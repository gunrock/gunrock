// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dobfs_enactor.cuh
 *
 * @brief Direction Optimal BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/dobfs/dobfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>
#include <gunrock/app/dobfs/dobfs_functor.cuh>


namespace gunrock {
namespace app {
namespace dobfs {

/**
 * @brief DOBFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class DOBFSEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime vertex_map_kernel_stats;

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the BFS search
     */
    long long           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for DOBFS kernel call. Must be called prior to each DOBFS search.
     *
     * @param[in] problem DOBFS Problem object which holds the graph data and DOBFS problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] vertex_map_grid_size CTA occupancy for vertex mapping kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int edge_map_grid_size,
        int vertex_map_grid_size)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        do {
            //initialize the host-mapped "done"
            if (!done) {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            //initialize runtime stats
            if (retval = edge_map_kernel_stats.Setup(edge_map_grid_size)) break;
            if (retval = vertex_map_kernel_stats.Setup(vertex_map_grid_size)) break;

            //Reset statistics
            iteration           = 0;
            total_runtimes      = 0;
            total_lifetimes     = 0;
            total_queued        = 0;
            done[0]             = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            column_indices_desc,
                            graph_slice->edges * sizeof(VertexId)),
                        "BFSEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief DOBFSEnactor constructor
     */
    DOBFSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief DOBFSEnactor destructor
     */
    virtual ~DOBFSEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "DOBFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "DOBFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        search_depth = this->iteration;

        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a direction optimal breadth-first search computing on the specified graph. (now only reverse bfs for testing purpose)
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam DOBFSProblem BFS Problem type.
     *
     * @param[in] problem DOBFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for DOBFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename DOBFSProblem>
    cudaError_t EnactDOBFS(
    DOBFSProblem                          *problem,
    typename DOBFSProblem::VertexId       src,
    int                                   max_grid_size = 0)
    {
        typedef typename DOBFSProblem::SizeT      SizeT;
        typedef typename DOBFSProblem::VertexId   VertexId;

        // Functors for reverse BFS
        typedef PrepareUnvisitedQueueFunctor<
            VertexId,
            SizeT,
            DOBFSProblem> UnvisitedQueueFunctor;

        typedef PrepareInputFrontierMapFunctor<
            VertexId,
            SizeT,
            DOBFSProblem> InputFrontierFunctor;

        typedef ReverseBFSFunctor<
            VertexId,
            SizeT,
            DOBFSProblem> RBFSFunctor;

        typedef SwitchToNormalFunctor<
            VertexId,
            SizeT,
            DOBFSProblem> SwitchFunctor;

        // Functors for BFS
        typedef gunrock::app::bfs::BFSFunctor<
            VertexId,
            SizeT,
            DOBFSProblem> BfsFunctor;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);

            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("BFS edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                printf("BFS vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }
            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename DOBFSProblem::GraphSlice   *graph_slice = problem->graph_slices[0];
            typename DOBFSProblem::DataSlice    *data_slice = problem->d_data_slices[0];
            SizeT num_unvisited_nodes = graph_slice->nodes - 1;
            SizeT current_frontier_size = 1;

            // Normal BFS
            {

            SizeT queue_length          = 1;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = 1;

            bool queue_reset = true; 


            fflush(stdout);
            // Step through BFS iterations
            
            while (done[0] < 0) {

                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, DOBFSProblem, BfsFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    this->edge_map_kernel_stats);


                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates 


                queue_index++;
                selector ^= 1;
                
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    printf(", %lld", (long long) queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = edge_map_kernel_stats.Accumulate(
                        edge_map_grid_size,
                        total_runtimes,
                        total_lifetimes)) break;
                }

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, DOBFSProblem, BfsFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;
                iteration++;
                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;

                if (INSTRUMENT || DEBUG) {
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                            vertex_map_grid_size,
                            total_runtimes,
                            total_lifetimes)) break;
                    }
                }

                num_unvisited_nodes -= queue_length;
                current_frontier_size = queue_length;
                if (num_unvisited_nodes < current_frontier_size*problem->alpha)
                    break;

                // Check if done
                if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) iteration);

            }

            if (retval) break;
            }
            if (DEBUG) printf("iter: %lld\n, alpha %f\n", iteration, problem->alpha);
              
            // Reverse BFS
            if (done[0] < 0) {
            if (DEBUG) printf("in RBFS.\n");

            //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[0], graph_slice->nodes);
            SizeT queue_length          = current_frontier_size;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = queue_length;

            bool queue_reset = true;
            
            // Prepare unvisited queue
            gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, DOBFSProblem, InputFrontierFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                        queue_reset,
                        queue_index,
                        1,
                        num_elements,
                        d_done,
                        graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                        graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                        data_slice,
                        work_progress,
                        graph_slice->frontier_elements[selector],           // max_in_queue
                        graph_slice->frontier_elements[selector^1],         // max_out_queue
                        this->vertex_map_kernel_stats);
            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_prepare_input_frontier::Kernel failed", __FILE__, __LINE__))) break;
            
            queue_length            = graph_slice->nodes;
            queue_index             = 0;        // Work queue index
            selector                = 0;
            num_elements          = graph_slice->nodes;

            gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, DOBFSProblem, UnvisitedQueueFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                        queue_reset,
                        queue_index,
                        1,
                        num_elements,
                        d_done,
                        problem->data_slices[0]->d_index_queue,             // d_in_queue
                        graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                        data_slice,
                        work_progress,
                        graph_slice->frontier_elements[selector],           // max_in_queue
                        graph_slice->frontier_elements[selector^1],         // max_out_queue
                        this->vertex_map_kernel_stats);
            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_prepare_unvisited_queue::Kernel failed", __FILE__, __LINE__))) break;

            queue_index++;
            selector ^= 1;
            if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
            num_elements = queue_length;

            // Now the unvisited queue is frontier_queues.d_keys[0], frontier_map_in and frontier_map_out are both ready too
            // Start Reverse BFS

            //util::DisplayDeviceResults(problem->data_slices[0]->d_frontier_map_in, graph_slice->nodes);

            SizeT last_queue_length = 0;
            while (done[0] < 0) {
                if (last_queue_length == queue_length) break;
                last_queue_length = queue_length;

                //util::DisplayDeviceResults(problem->graph_slices[0]->frontier_queues.d_keys[selector], queue_length);

                if (selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<EdgeMapPolicy, DOBFSProblem, RBFSFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    problem->data_slices[0]->d_index_queue,            // d_in_index_queue
                    problem->data_slices[0]->d_frontier_map_in,
                    problem->data_slices[0]->d_frontier_map_out,
                    problem->data_slices[0]->d_col_offsets,
                    problem->data_slices[0]->d_row_indices,
                    data_slice,
                    this->work_progress,
                    this->edge_map_kernel_stats);
                } else {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<EdgeMapPolicy, DOBFSProblem, RBFSFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    problem->data_slices[0]->d_index_queue,            // d_in_index_queue
                    problem->data_slices[0]->d_frontier_map_out,
                    problem->data_slices[0]->d_frontier_map_in,
                    problem->data_slices[0]->d_col_offsets,
                    problem->data_slices[0]->d_row_indices,
                    data_slice,
                    this->work_progress,
                    this->edge_map_kernel_stats);
                }


                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates  
                

                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    printf(", %lld", (long long) queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = edge_map_kernel_stats.Accumulate(
                        edge_map_grid_size,
                        total_runtimes,
                        total_lifetimes)) break;
                }

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, DOBFSProblem, RBFSFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;
                iteration++;

                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                if (INSTRUMENT || DEBUG) {
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                            vertex_map_grid_size,
                            total_runtimes,
                            total_lifetimes)) break;
                    }
                }
                if (queue_length < graph_slice->nodes/problem->beta) break;

                // Check if done
                if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) iteration);

            }

            if (retval) break;

            }

            if (DEBUG) printf("iter: %lld\n, beta %f\n", iteration, problem->beta);

            // Normal BFS
            if (done[0] < 0) {
                if (DEBUG) printf("back to normal BFS.\n");
            SizeT queue_length          = graph_slice->nodes;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = queue_length;

            bool queue_reset = true;

            gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, DOBFSProblem, SwitchFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                        queue_reset,
                        queue_index,
                        1,
                        num_elements,
                        d_done,
                        problem->data_slices[0]->d_index_queue,             // d_in_queue
                        graph_slice->frontier_queues.d_keys[selector],    // d_out_queue
                        data_slice,
                        work_progress,
                        graph_slice->frontier_elements[selector],           // max_in_queue
                        graph_slice->frontier_elements[selector^1],         // max_out_queue
                        this->vertex_map_kernel_stats);
            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_switch_to_normal::Kernel failed", __FILE__, __LINE__))) break;

            queue_index++;
            if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;

            // Step through BFS iterations
            num_elements = queue_length;
            queue_index = 0;
            
            while (done[0] < 0) {

                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, DOBFSProblem, BfsFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    this->edge_map_kernel_stats);


                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates 


                queue_index++;
                selector ^= 1;
                
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    printf(", %lld", (long long) queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = edge_map_kernel_stats.Accumulate(
                        edge_map_grid_size,
                        total_runtimes,
                        total_lifetimes)) break;
                }

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, DOBFSProblem, BfsFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;
                iteration++;
                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;

                if (INSTRUMENT || DEBUG) {
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                            vertex_map_grid_size,
                            total_runtimes,
                            total_lifetimes)) break;
                    }
                }
                
                // Check if done
                if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) iteration);

            }
            if (retval) break;
            }
            
        } while(0);

        if (DEBUG) printf("\nGPU BFS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename DOBFSProblem>
    cudaError_t Enact(
        DOBFSProblem                      *problem,
        typename DOBFSProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                DOBFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                8>                                  // LOG_SCHEDULE_GRANULARITY
                VertexMapPolicy;
                
                typedef gunrock::oprtr::edge_map_backward::KernelPolicy<
                DOBFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7>                                  // LOG_SCHEDULE_GRANULARITY
                EdgeMapPolicy;

                return EnactDOBFS<EdgeMapPolicy, VertexMapPolicy, DOBFSProblem>(
                problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace dobfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
