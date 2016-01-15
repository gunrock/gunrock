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

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

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
template <typename _Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK>
class DOBFSEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

    // Members
    protected:

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    //volatile int        *done;
    //int                 *d_done;
    //cudaEvent_t         throttle_event;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for DOBFS kernel call. Must be called prior to each DOBFS search.
     *
     * @param[in] problem DOBFS Problem object which holds the graph data and DOBFS problem data to compute.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;

        cudaError_t retval = cudaSuccess;

        do {
            //initialize the host-mapped "done"
            //if (!done)
            {
                //int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                //if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                //    "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                //if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                //    "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                //if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                //    "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            //done[0]             = -1;

            //graph slice
            GraphSlice<SizeT, VertexId, Value>
                                             *graph_slice = problem->graph_slices[0];
            typename ProblemData::DataSlice  *data_slice  = problem->data_slices[0];

            // Bind row-offsets texture
            /*cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;
            */

            if (ProblemData::ENABLE_IDEMPOTENCE) {
                int bytes = (graph_slice->nodes + 8 - 1) / 8;
                cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();

                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref.channelDesc = bitmask_desc;
                if (retval = util::GRError(cudaBindTexture(
                                0,
                                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,
                                data_slice->d_visited_mask,
                                bytes),
                            "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
            }

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            graph_slice->edges * sizeof(VertexId)),
                        "BFSEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);

        return retval;
    }

    public:

    /**
     * @brief DOBFSEnactor constructor
     */
    DOBFSEnactor(int *gpu_idx):
        EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>(EDGE_FRONTIERS, 1, gpu_idx)
        //done(NULL),
        //d_done(NULL)
    {}

    /**
     * @brief DOBFSEnactor destructor
     */
    virtual ~DOBFSEnactor()
    {
        //if (done) {
            //util::GRError(cudaFreeHost((void*)done),
            //    "DOBFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            //util::GRError(cudaEventDestroy(throttle_event),
            //    "DOBFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        //}
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @ brief Obtain statistics about the last BFS search enacted.
     *
     * @ param[out] total_queued Total queued elements in BFS kernel running.
     * @ param[out] search_depth Search depth of BFS algorithm.
     * @ param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     * spaces between @ and name are to eliminate doxygen warnings
     */
    /*template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId  &search_depth,
        double    &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->enactor_stats->total_queued[0];
        search_depth = this->enactor_stats->iteration;

        avg_duty = (this->enactor_stats->total_lifetimes >0) ?
            double(this->enactor_stats->total_runtimes) / this->enactor_stats->total_lifetimes : 0.0;
    }*/

    /** @} */

    /**
     * @brief Enacts a direction optimal breadth-first search computing on the specified graph. (now only reverse bfs for testing purpose)
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam BackwardAdvanceKernelPolicy Kernel policy for backward advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam DOBFSProblem BFS Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem DOBFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for DOBFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename BackwardAdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BackwardFilterKernelPolicy,
        typename DOBFSProblem>
    cudaError_t EnactDOBFS(
    ContextPtr                            context,
    DOBFSProblem                         *problem,
    typename DOBFSProblem::VertexId       src,
    int                                   max_grid_size = 0)
    {
        typedef typename DOBFSProblem::SizeT      SizeT;
        typedef typename DOBFSProblem::VertexId   VertexId;

        // Functors for reverse BFS
        typedef PrepareUnvisitedQueueFunctor<
            VertexId,
            SizeT,
            VertexId,
            DOBFSProblem> UnvisitedQueueFunctor;

        typedef PrepareInputFrontierMapFunctor<
            VertexId,
            SizeT,
            VertexId,
            DOBFSProblem> InputFrontierFunctor;

        typedef ReverseBFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            DOBFSProblem> RBFSFunctor;

        typedef SwitchToNormalFunctor<
            VertexId,
            SizeT,
            VertexId,
            DOBFSProblem> SwitchFunctor;

        // Functors for BFS
        typedef gunrock::app::bfs::BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            DOBFSProblem> BfsFunctor;

        FrontierAttribute<SizeT>
                     *frontier_attribute = &this->frontier_attribute[0];
        EnactorStats *enactor_stats      = &this->enactor_stats     [0];
        typename DOBFSProblem::DataSlice
                     *data_slice         =  problem->data_slices    [0];
        util::DoubleBuffer<SizeT, VertexId, Value>
                     *frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime
                     *work_progress      = &this->work_progress     [0];
        cudaStream_t  stream             =  data_slice->streams[0];
        cudaError_t   retval             = cudaSuccess;
        SizeT *d_scanned_edges    = NULL;

        do {
            // Determine grid size(s)
            if (DEBUG) {
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }
            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY)) break;

            // Single-gpu graph slice
            GraphSlice<SizeT, VertexId, Value> *graph_slice  = problem->graph_slices[0];
            typename DOBFSProblem::DataSlice   *d_data_slice = problem->d_data_slices[0];
            SizeT num_unvisited_nodes = graph_slice->nodes - 1;
            SizeT current_frontier_size = 1;

            //if (retval = util::GRError(cudaMalloc(
            //                (void**)&d_scanned_edges,
            //                graph_slice->edges * sizeof(SizeT)),
            //            "PBFSProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            if (retval = data_slice -> scanned_edges[0].EnsureSize(graph_slice->edges))
                return retval;
            d_scanned_edges = data_slice -> scanned_edges[0].GetPointer(util::DEVICE);

            // Normal BFS
            {

                frontier_attribute->queue_length         = 1;
                frontier_attribute->queue_index          = 0;        // Work queue index
                frontier_attribute->selector             = 0;
                frontier_attribute->queue_reset          = true;


                fflush(stdout);
                // Step through BFS iterations

                //while (done[0] < 0) {
                while (frontier_attribute->queue_length > 0) {

                    enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
                    // Edge Map
                    gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, DOBFSProblem, BfsFunctor>(
                        //d_done,
                        enactor_stats[0],
                        frontier_attribute[0],
                        d_data_slice,
                        (VertexId*)NULL,
                        (bool*    )NULL,
                        (bool*    )NULL,
                        d_scanned_edges,
                        frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                        frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                        (VertexId*)NULL,
                        frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                        graph_slice->row_offsets   .GetPointer(util::DEVICE),
                        graph_slice->column_indices.GetPointer(util::DEVICE),
                        (SizeT*   )NULL,
                        (VertexId*)NULL,
                        graph_slice->nodes, //graph_slice->frontier_elements[frontier_attribute.selector],   // max_in_queue
                        graph_slice->edges, //graph_slice->frontier_elements[frontier_attribute.selector^1], // max_out_queue
                        work_progress[0],
                        context[0],
                        stream,
                        gunrock::oprtr::advance::V2V,
                        false,
                        false,
                        true);


                    // Only need to reset queue for once
                    if (frontier_attribute->queue_reset)
                        frontier_attribute->queue_reset = false;

                    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "Advance::LaunchKernel failed", __FILE__, __LINE__))) break;
                    //cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates


                    frontier_attribute->queue_index++;
                    frontier_attribute->selector ^= 1;

                    if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                    enactor_stats -> edges_queued[0] += frontier_attribute -> queue_length;

                    if (DEBUG) {
                        //if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                        printf(", %lld", (long long) frontier_attribute->queue_length);
                    }

                    if (INSTRUMENT) {
                        if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                                    enactor_stats->advance_grid_size,
                                    enactor_stats->total_runtimes,
                                    enactor_stats->total_lifetimes)) break;
                    }

                    // Throttle
                    /*if (enactor_stats->iteration & 1) {
                        if (retval = util::GRError(cudaEventRecord(throttle_event),
                                    "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                    } else {
                        if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                                    "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                    }*/

                    // Check if done
                    //if (done[0] == 0) break;
                    if (frontier_attribute->queue_length == 0) break;

                    // Filter
                    gunrock::oprtr::filter::LaunchKernel
                        <FilterKernelPolicy, DOBFSProblem, BfsFunctor>(
                        enactor_stats->filter_grid_size, 
                        FilterKernelPolicy::THREADS,
                        0, 0,
                        enactor_stats->iteration + 1,
                        frontier_attribute->queue_reset,
                        frontier_attribute->queue_index,
                        //enactor_stats.num_gpus,
                        frontier_attribute->queue_length,
                        //d_done,
                        frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                        frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE),    // d_pred_in_queue
                        frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                        d_data_slice,
                        data_slice->d_visited_mask,
                        work_progress[0],
                        frontier_queue->keys  [frontier_attribute->selector  ].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                        frontier_queue->keys  [frontier_attribute->selector^1].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                        enactor_stats->filter_kernel_stats);

                    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                    //cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                    frontier_attribute->queue_index++;
                    frontier_attribute->selector ^= 1;

                    if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;

                    if (INSTRUMENT || DEBUG) {
                        //enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
                        if (DEBUG) printf(", %lld", (long long) frontier_attribute->queue_length);
                        if (INSTRUMENT) {
                            if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                                        enactor_stats->filter_grid_size,
                                        enactor_stats->total_runtimes,
                                        enactor_stats->total_lifetimes)) break;
                        }
                    }

                    num_unvisited_nodes -= frontier_attribute->queue_length;
                    current_frontier_size = frontier_attribute->queue_length;
                    enactor_stats->iteration++;
                    if (num_unvisited_nodes < current_frontier_size*problem->alpha) break;

                    // Check if done
                    //if (done[0] == 0) break;
                    if (frontier_attribute->queue_length == 0) break;

                    if (DEBUG) printf("\n%lld", (long long) enactor_stats->iteration);

                }

                if (retval) break;
            }
            if (DEBUG) printf("iter: %lld\n, alpha %f\n", enactor_stats->iteration, problem->alpha);

            // Reverse BFS
            //if (done[0] < 0) {
            if (frontier_attribute->queue_length != 0) {
                if (DEBUG) { printf("in RBFS.\n"); }

            //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[0], graph_slice->nodes);
            frontier_attribute->queue_length         = current_frontier_size;
            frontier_attribute->queue_index          = 0;        // Work queue index
            frontier_attribute->selector             = 0;
            frontier_attribute->queue_reset          = true;

            // Prepare unvisited queue
            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, DOBFSProblem, InputFrontierFunctor>(
                enactor_stats->filter_grid_size, 
                FilterKernelPolicy::THREADS,
                0, 0,
                -1,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                //enactor_stats.num_gpus,
                frontier_attribute->queue_length,
                //d_done,
                frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                (VertexId*)NULL,
                frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->keys  [frontier_attribute->selector  ].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                frontier_queue->keys  [frontier_attribute->selector^1].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                enactor_stats->filter_kernel_stats);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_prepare_input_frontier::Kernel failed", __FILE__, __LINE__))) break;

            frontier_attribute->queue_length            = graph_slice->nodes;
            frontier_attribute->queue_index             = 0;        // Work queue index
            frontier_attribute->selector                = 0;

            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, DOBFSProblem, UnvisitedQueueFunctor>(
                enactor_stats->filter_grid_size, 
                FilterKernelPolicy::THREADS,
                0, 0,
                -1,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                //enactor_stats.num_gpus,
                frontier_attribute->queue_length,
                //d_done,
                data_slice->d_index_queue,             // d_in_queue
                (VertexId*)NULL,
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_prepare_unvisited_queue::Kernel failed", __FILE__, __LINE__))) break;

            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;
            if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;

            // Now the unvisited queue is frontier_queues.d_keys[0], frontier_map_in and frontier_map_out are both ready too
            // Start Reverse BFS

            SizeT last_queue_length = 0;
            //while (done[0] < 0) {
            while (frontier_attribute->queue_length != 0) {
                if (last_queue_length == frontier_attribute->queue_length)
                {
                    //done[0] = 0;
                    frontier_attribute->queue_length = 0;
                    break;
                }
                last_queue_length = frontier_attribute->queue_length;

                enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<BackwardAdvanceKernelPolicy, DOBFSProblem, RBFSFunctor>(
                    //d_done,
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    data_slice->d_index_queue,
                    data_slice->d_frontier_map_in,
                    data_slice->d_frontier_map_out,
                    d_scanned_edges,
                    frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),              // d_in_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->column_offsets.GetPointer(util::DEVICE),
                    graph_slice->row_indices   .GetPointer(util::DEVICE),
                    graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1],                   // max_out_queue
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    false,
                    true);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_backward::Kernel failed", __FILE__, __LINE__))) break;
                //cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates

                //util::DisplayDeviceResults(problem->data_slices[0]->d_frontier_map_out, graph_slice->nodes);

                if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                if (DEBUG) {
                    //if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                    printf(", %lld", (long long) frontier_attribute->queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                        enactor_stats->advance_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes)) break;
                }

                // Throttle
                /*if (enactor_stats->iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }*/

                // Check if done
                //if (done[0] == 0) break;
                if (frontier_attribute->queue_length == 0) break;

                enactor_stats -> edges_queued[0] += frontier_attribute -> queue_length;
                // Vertex Map
                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, DOBFSProblem, RBFSFunctor>(
                    enactor_stats->filter_grid_size, 
                    FilterKernelPolicy::THREADS,
                    0, 0,
                    -1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    //enactor_stats.num_gpus,
                    frontier_attribute->queue_length,
                    //d_done,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                    (VertexId*)NULL,
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                    d_data_slice,
                    NULL,
                    work_progress[0],
                    frontier_queue->keys[frontier_attribute->selector  ].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats->filter_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                //cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                enactor_stats->iteration++;

                if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                if (INSTRUMENT || DEBUG) {
                    //enactor_stats->total_queued[0] += frontier_attribute->queue_length;
                    if (DEBUG) printf(", %lld", (long long) frontier_attribute->queue_length);
                    if (INSTRUMENT) {
                        if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                            enactor_stats->filter_grid_size,
                            enactor_stats->total_runtimes,
                            enactor_stats->total_lifetimes)) break;
                    }
                }
                if (frontier_attribute->queue_length < graph_slice->nodes/problem->beta) break;

                // Check if done
                //if (done[0] == 0) break;
                if (frontier_attribute->queue_length == 0) break;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats->iteration);

            }

            if (retval) break;

            }

            if (DEBUG) printf("iter: %lld\n, beta %f\n", enactor_stats->iteration, problem->beta);

            // Normal BFS
            //if (done[0] < 0) {
            if (frontier_attribute -> queue_length != 0) {
                if (DEBUG) printf("back to normal BFS.\n");

            //If selector == 1, copy map_in to map_out
            if (frontier_attribute->selector == 1) {
                if (retval = util::GRError(cudaMemcpy(
                    data_slice->d_frontier_map_out,
                    data_slice->d_frontier_map_in,
                    graph_slice->nodes*sizeof(bool),
                    cudaMemcpyDeviceToDevice),
                    "DOBFS cudaMemcpy frontier_map_in to frontier_map_out failed", __FILE__, __LINE__)) break;
            }

            frontier_attribute->queue_length         = graph_slice->nodes;
            //frontier_attribute->queue_length         = 1;
            frontier_attribute->queue_index          = 0;        // Work queue index
            frontier_attribute->selector             = 0;
            frontier_attribute->queue_reset          = true;

            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, DOBFSProblem, SwitchFunctor>(
                enactor_stats->filter_grid_size, 
                FilterKernelPolicy::THREADS,
                0, 0,
                -1,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                //enactor_stats.num_gpus,
                frontier_attribute->queue_length,
                //d_done,
                data_slice->d_index_queue,             // d_in_queue
                NULL,
                frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),    // d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                enactor_stats->filter_kernel_stats);
            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_switch_to_normal::Kernel failed", __FILE__, __LINE__))) break;

            frontier_attribute->queue_index++;
            if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;

            // Step through BFS iterations
            frontier_attribute->queue_index = 0;

            //while (done[0] < 0) {
            while (frontier_attribute -> queue_length !=0) {

                enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, DOBFSProblem, BfsFunctor>(
                    //d_done,
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    d_scanned_edges,
                    frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),          // d_out_queue
                    (VertexId*)NULL,
                    frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    false,
                    true);

                // Only need to reset queue for once
                if (frontier_attribute->queue_reset)
                    frontier_attribute->queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                //cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates


                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;

                if (DEBUG) {
                    //if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    printf(", %lld", (long long) frontier_attribute->queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                        enactor_stats->advance_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes)) break;
                }

                // Throttle
                /*if (enactor_stats->iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }*/

                // Check if done
                //if (done[0] == 0) break;
                if (frontier_attribute -> queue_length == 0) break;
                enactor_stats->edges_queued[0] += frontier_attribute->queue_length;

                // Vertex Map
                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, DOBFSProblem, BfsFunctor>(
                    enactor_stats->filter_grid_size, 
                    FilterKernelPolicy::THREADS,
                    0, 0,
                    enactor_stats->iteration + 1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    //enactor_stats.num_gpus,
                    frontier_attribute->queue_length,
                    //d_done,
                    frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                    frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE),    // d_pred_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                    d_data_slice,
                    data_slice->d_visited_mask,
                    work_progress[0],
                    frontier_queue->keys  [frontier_attribute->selector  ].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    frontier_queue->keys  [frontier_attribute->selector^1].GetSize(),//graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats->filter_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                //cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                enactor_stats->iteration++;
                if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;

                if (INSTRUMENT || DEBUG) {
                    //enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
                    if (DEBUG) printf(", %lld", (long long) frontier_attribute->queue_length);
                    if (INSTRUMENT) {
                        if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                            enactor_stats->filter_grid_size,
                            enactor_stats->total_runtimes,
                            enactor_stats->total_lifetimes)) break;
                    }
                }

                // Check if done
                //if (done[0] == 0) break;
                if (frontier_attribute -> queue_length == 0) break;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats->iteration);

            }
            if (retval) break;
            }

        } while(0);
        //if (d_scanned_edges) cudaFree(d_scanned_edges);

        if (DEBUG) printf("\nGPU BFS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Direction Optimal BFS Enact kernel entry.
     *
     * @tparam DOBFSProblem DOBFS Problem type. @see DOBFSProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem Pointer to DOBFSProblem object.
     * @param[in] src Source node for DOBFS.
     * @param[in] max_grid_size Max grid size for DOBFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename DOBFSProblem>
    cudaError_t Enact(
        ContextPtr                         context,
        DOBFSProblem                      *problem,
        typename DOBFSProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

            if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    DOBFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    0,                                  // SATURATION QUIT
                    true,                               // DEQUEUE_PROBLEM_SIZE
                    8,                                  // MIN_CTA_OCCUPANCY
                    8,                                  // LOG_THREADS
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    5,                                  // END_BITMASK_CULL
                    8>                                  // LOG_SCHEDULE_GRANULARITY
                        FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    DOBFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,
                    32*128,
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                                 // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB_LIGHT>
                        AdvanceKernelPolicy;

                typedef gunrock::oprtr::filter::KernelPolicy<
                    DOBFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    0,                                  // SATURATION QUIT
                    true,                               // DEQUEUE_PROBLEM_SIZE
                    8,                                  // MIN_CTA_OCCUPANCY
                    8,                                  // LOG_THREADS
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    5,                                  // END_BITMASK_CULL
                    8>                                  // LOG_SCHEDULE_GRANULARITY
                        BackwardFilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    DOBFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                             // LIGHT_EDGE_THRESHOLD
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                                 // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB_BACKWARD>
                        BackwardAdvanceKernelPolicy;


                if (DOBFSProblem::ENABLE_IDEMPOTENCE) {
                return EnactDOBFS<AdvanceKernelPolicy, BackwardAdvanceKernelPolicy, FilterKernelPolicy, BackwardFilterKernelPolicy, DOBFSProblem>(
                        context, problem, src, max_grid_size);
                } else {
                return EnactDOBFS<AdvanceKernelPolicy, BackwardAdvanceKernelPolicy, BackwardFilterKernelPolicy, BackwardFilterKernelPolicy, DOBFSProblem>(
                        context, problem, src, max_grid_size);
                }
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
