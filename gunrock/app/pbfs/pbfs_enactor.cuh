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
 * @brief PBFS Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/pbfs/pbfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Partitioned BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class PBFSEnactor : public EnactorBase
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
                    "PBFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "PBFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "PBFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
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
            
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief PBFSEnactor constructor
     */
    PBFSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief PBFSEnactor destructor
     */
    virtual ~PBFSEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "PBFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "PBFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
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
     * @tparam PBFSProblem PBFS Problem type.
     *
     * @param[in] problem PBFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename PBFSProblem>
    cudaError_t EnactPBFS(
    CudaContext                          &context,
    PBFSProblem                          *problem,
    typename PBFSProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        typedef typename PBFSProblem::SizeT      SizeT;
        typedef typename PBFSProblem::VertexId   VertexId;

        typedef BFSFunctor<
            VertexId,
            SizeT,
            PBFSProblem> BfsFunctor;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);
            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("Partitioned BFS edge map occupancy %d, level-grid size %d\n",
                edge_map_occupancy, edge_map_grid_size);
                printf("Partitioned BFS vertex map occupancy %d, level-grid size %d\n",
                vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }

            // Lazy initialization
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename PBFSProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename PBFSProblem::DataSlice *data_slice = problem->d_data_slices[0];


            unsigned int queue_length      = 1;
            VertexId queue_index    = 0;
            int selector            = 0;
            SizeT num_elements      = 1;

            bool queue_reset        = true;

            while (done[0] < 0) {

                //Partitioned Edge Map
                //
                // Get Rowoffsets
                // Use scan to compute edge_offsets for each vertex in the frontier
                // MarkPartitionSizes
                // Use sorted sort to compute partition bound for each work-chunk
                // load edge-expand-partitioned kernel
                int num_block = (queue_length + EdgeMapPolicy::THREADS - 1)/EdgeMapPolicy::THREADS;
                gunrock::oprtr::edge_map_partitioned::GetEdgeCounts<EdgeMapPolicy, PBFSProblem, BfsFunctor> <<< num_block, EdgeMapPolicy::THREADS >>>(
                                        graph_slice->d_row_offsets,
                                        graph_slice->frontier_queues.d_keys[selector],
                                       problem->data_slices[0]->d_scanned_edges,
                                        queue_length,
                                        graph_slice->frontier_elements[selector],
                                        graph_slice->frontier_elements[selector^1]);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_partitioned kernel failed", __FILE__, __LINE__))) break;
                //Scan<MgpuScanTypeInc>((unsigned int*)data_slice->d_scanned_edges, queue_length, context);
                Scan<MgpuScanTypeInc>((int*)problem->data_slices[0]->d_scanned_edges, queue_length, INT_MIN, mgpu::maximum<int>(),
		(int*)0, (int*)0, (int*)problem->data_slices[0]->d_scanned_edges, context);
                SizeT *temp = new SizeT[1];
                cudaMemcpy(temp, problem->data_slices[0]->d_scanned_edges+queue_length-1, sizeof(SizeT), cudaMemcpyDeviceToHost);
                SizeT output_queue_len = temp[0];
                
                // Edge Expand Kernel
                {
                    if (output_queue_len < EdgeMapPolicy::LIGHT_EDGE_THRESHOLD)
                    {
                        gunrock::oprtr::edge_map_partitioned::RelaxLightEdges<EdgeMapPolicy, PBFSProblem, BfsFunctor> <<< num_block, EdgeMapPolicy::THREADS >>>(
                                        queue_reset,
                                        queue_index,
                                        graph_slice->d_row_offsets,
                                        graph_slice->d_column_indices,
                                        problem->data_slices[0]->d_scanned_edges,
                                        d_done,
                                        graph_slice->frontier_queues.d_keys[selector],
                                        graph_slice->frontier_queues.d_keys[selector^1],
                                        data_slice,
                                        queue_length,
                                        output_queue_len,
                                        graph_slice->frontier_elements[selector],
                                        graph_slice->frontier_elements[selector^1],
                                        work_progress,
                                        this->edge_map_kernel_stats);
                    }
                    else
                    {
                        unsigned int split_val = (output_queue_len + EdgeMapPolicy::BLOCKS - 1) / EdgeMapPolicy::BLOCKS;
                        num_block = (EdgeMapPolicy::BLOCKS + EdgeMapPolicy::THREADS - 1)/EdgeMapPolicy::THREADS;
                        gunrock::oprtr::edge_map_partitioned::MarkPartitionSizes<EdgeMapPolicy, PBFSProblem, BfsFunctor> <<< num_block, EdgeMapPolicy::THREADS >>>(
                                        problem->data_slices[0]->d_node_locks,
                                        split_val,
                                        EdgeMapPolicy::BLOCKS);
                        SortedSearch<MgpuBoundsLower>(problem->data_slices[0]->d_node_locks, EdgeMapPolicy::BLOCKS, problem->data_slices[0]->d_scanned_edges, queue_length, problem->data_slices[0]->d_node_locks, context);

                         gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges<EdgeMapPolicy, PBFSProblem, BfsFunctor> <<< EdgeMapPolicy::BLOCKS, EdgeMapPolicy::THREADS >>>(
                                        queue_reset,
                                        queue_index,
                                        graph_slice->d_row_offsets,
                                        graph_slice->d_column_indices,
                                        problem->data_slices[0]->d_scanned_edges,
                                        problem->data_slices[0]->d_node_locks,
                                        EdgeMapPolicy::BLOCKS,
                                        d_done,
                                        graph_slice->frontier_queues.d_keys[selector],
                                        graph_slice->frontier_queues.d_keys[selector^1],
                                        data_slice,
                                        queue_length,
                                        output_queue_len,
                                        split_val,
                                        graph_slice->frontier_elements[selector],
                                        graph_slice->frontier_elements[selector^1],
                                        work_progress,
                                        this->edge_map_kernel_stats);   

                    }
                }

                //Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_partitioned kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); //give host memory mapped visibility to GPU updates

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
                    if (retval = util::GRError(cudaEventRecord(throttle_event), "PBFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event), "PBFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, PBFSProblem, BfsFunctor>
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

        } while(0);

        if (DEBUG) printf("\nGPU BFS Done.\n");
        return retval;
    }

        /**
         * \addtogroup PublicInterface
         * @{
         */

    /**
     * @brief Partitioned BFS Enact kernel entry.
     *
     * @tparam PBFSProblem BFS Problem type. @see PBFSProblem
     *
     * @param[in] problem Pointer to PBFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename PBFSProblem>
    cudaError_t Enact(
        CudaContext                      &context,
        PBFSProblem                      *problem,
        typename PBFSProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
       // Define Kernel Policy
       // Load Enactor
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                PBFSProblem,                         // Problem data type
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

                typedef gunrock::oprtr::edge_map_partitioned::KernelPolicy<
                PBFSProblem,                        // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32 * 1024>                          // LIGHT_EDGE_THRESHOLD
                EdgeMapPolicy;

                return EnactPBFS<EdgeMapPolicy, VertexMapPolicy, PBFSProblem>(
                context, problem, src, max_grid_size);
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
