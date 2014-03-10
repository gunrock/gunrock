// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>


namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class BFSEnactor : public EnactorBase
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
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for BFS kernel call. Must be called prior to each BFS search.
     *
     * @param[in] problem BFS Problem object which holds the graph data and BFS problem data to compute.
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
            typename ProblemData::DataSlice *data_slice = problem->data_slices[0];

            // Bind row-offsets and bitmask texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            if (ProblemData::ENABLE_IDEMPOTENCE) {
                int bytes = (graph_slice->nodes + 8 - 1) / 8;
                cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();
                if (retval = util::GRError(cudaBindTexture(
                                0,
                                gunrock::oprtr::vertex_map::BitmaskTex<unsigned char>::ref,
                                data_slice->d_visited_mask,
                                bitmask_desc,
                                bytes),
                            "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
            }

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
     * @brief BFSEnactor constructor
     */
    BFSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "BFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "BFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
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
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BFSProblem>
    cudaError_t EnactBFS(
    BFSProblem                          *problem,
    typename BFSProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;

        typedef BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            BFSProblem> BfsFunctor;

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


            // Lazy initialization
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename BFSProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename BFSProblem::DataSlice *data_slice = problem->d_data_slices[0];

            SizeT queue_length          = 1;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = 1;

            bool queue_reset = true; 


            fflush(stdout);
            // Step through BFS iterations
            
            VertexId *h_cur_queue = new VertexId[graph_slice->edges];
            while (done[0] < 0) {

                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, BFSProblem, BfsFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_values[selector^1],          // d_pred_out_queue
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
                    util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], queue_length);
                    util::DisplayDeviceResults(graph_slice->frontier_queues.d_values[selector], queue_length);

                    /*if (iteration == 2) {
                        cudaMemcpy(h_cur_queue, graph_slice->frontier_queues.d_keys[selector], sizeof(VertexId)*queue_length, cudaMemcpyDeviceToHost);
                        int neg_num = 0;
                        std::sort(h_cur_queue, h_cur_queue + queue_length);
                        for (int i = 0; i < queue_length; ++i)
                        {
                            if (h_cur_queue[i] == -1)
                                neg_num++;
                        }
                        printf("(%d)", neg_num);
                    }*/
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
		printf("Before Vertex Mapping:");
                util::DisplayDeviceResults(problem->data_slices[0]->d_labels, graph_slice->nodes);
		// Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, BFSProblem, BfsFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    iteration+1,
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    data_slice,
                    problem->data_slices[0]->d_visited_mask,
                    work_progress,
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    this->vertex_map_kernel_stats);
                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates
		
		util::DisplayDeviceResults(problem->data_slices[0]->d_labels, graph_slice->nodes);
                
		queue_index++;
                selector ^= 1;
                iteration++;

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
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

            delete[] h_cur_queue;
            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
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
    template <typename BFSProblem>
    cudaError_t Enact(
        BFSProblem                      *problem,
        typename BFSProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        if (BFSProblem::ENABLE_IDEMPOTENCE) {
            if (this->cuda_props.device_sm_version >= 300) {
                typedef gunrock::oprtr::vertex_map::KernelPolicy<
                    BFSProblem,                         // Problem data type
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
                    VertexMapPolicy;

                typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    8,                                  // LOG_THREADS
                    0,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7>                                  // LOG_SCHEDULE_GRANULARITY
                        EdgeMapPolicy;

                return EnactBFS<EdgeMapPolicy, VertexMapPolicy, BFSProblem>(
                        problem, src, max_grid_size);
            }
        } else {
                if (this->cuda_props.device_sm_version >= 300) {
                typedef gunrock::oprtr::vertex_map::KernelPolicy<
                    BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    VertexMapPolicy;

                typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    6,                                  // LOG_THREADS
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7>                                  // LOG_SCHEDULE_GRANULARITY
                        EdgeMapPolicy;

                return EnactBFS<EdgeMapPolicy, VertexMapPolicy, BFSProblem>(
                        problem, src, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
