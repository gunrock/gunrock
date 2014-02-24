// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_enactor.cuh
 *
 * @brief HITS (Hyperlink-Induced Topic Search) Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/hits/hits_problem.cuh>
#include <gunrock/app/hits/hits_functor.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace hits {

/**
 * @brief HITS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class HITSEnactor : public EnactorBase
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
     * Current iteration, also used to get the final search depth of the HITS search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for HITS kernel call. Must be called prior to each HITS search.
     *
     * @param[in] problem HITS Problem object which holds the graph data and HITS problem data to compute.
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
                    "HITSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "HITSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "HITSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
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
     * @brief HITSEnactor constructor
     */
    HITSEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief HITSEnactor destructor
     */
    virtual ~HITSEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "HITSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "HITSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }
    template <typename ProblemData>
    void NormalizeRank(ProblemData *problem, CudaContext &context, int hub_or_auth, int nodes)
    {

        typedef typename ProblemData::Value         Value;
        Value *rank_curr;
        Value *rank_next;
        if (hub_or_auth == 0) {
            rank_curr = problem->data_slices[0]->d_hrank_curr;
            rank_next = problem->data_slices[0]->d_hrank_next;
            //printf("hub\n");
        } else {
            rank_curr = problem->data_slices[0]->d_arank_curr;
            rank_next = problem->data_slices[0]->d_arank_next;
            //printf("auth\n");
        }

        //swap rank_curr and rank_next
        util::MemsetCopyVectorKernel<<<128, 128>>>(rank_curr, rank_next, nodes); 

        //Compute square
        util::MemsetMultiplyVectorKernel<<<128, 128>>>(rank_next, rank_next, nodes);

        //Compute sum by reduce
        Value accumulated_rank = Reduce(rank_next, nodes, context);
        //Compute invsqrt
        accumulated_rank = 1.0/sqrt(accumulated_rank);
        util::MemsetScaleKernel<<<128, 128>>>(rank_curr, accumulated_rank, nodes);

        util::MemsetKernel<<<128, 128>>>(rank_next, (Value)0.0, nodes);

        //util::DisplayDeviceResults(rank_curr, nodes);


    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last HITS search enacted.
     *
     * @param[out] total_queued Total queued elements in HITS kernel running.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a HITS computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam HITSProblem HITS Problem type.
     *
     * @param[in] problem HITSProblem object.
     * @param[in] max_iteration Max number of iterations of HITS algorithm
     * @param[in] max_grid_size Max grid size for HITS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename HITSProblem>
    cudaError_t EnactHITS(
    CudaContext                        &context,
    HITSProblem                          *problem,
    typename HITSProblem::SizeT           max_iteration,
    int                                 max_grid_size = 0)
    {
        typedef typename HITSProblem::SizeT       SizeT;
        typedef typename HITSProblem::VertexId    VertexId;
        typedef typename HITSProblem::Value       Value;

        typedef HUBFunctor<
            VertexId,
            SizeT,
            Value,
            HITSProblem> HubFunctor;

        typedef AUTHFunctor<
            VertexId,
            SizeT,
            Value,
            HITSProblem> AuthFunctor;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);

            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("HITS edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                printf("HITS vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }

            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename HITSProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename HITSProblem::DataSlice *data_slice = problem->d_data_slices[0];

            // Bind row-offsets texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>(); 

            SizeT queue_length          = graph_slice->nodes;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = graph_slice->nodes;

            bool queue_reset = true;
            int edge_map_queue_len = num_elements;

            // Step through HITS iterations 
            while (done[0] < 0) { 

                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "HITSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

                if (retval = work_progress.SetQueueLength(queue_index, edge_map_queue_len)) break;
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, HITSProblem, HubFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    NULL,
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    this->edge_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates  


                //util::DisplayDeviceResults(problem->data_slices[0]->d_hrank_next,graph_slice->nodes);

                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_column_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "HITSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

                if (retval = work_progress.SetQueueLength(queue_index, edge_map_queue_len)) break;
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, HITSProblem, AuthFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    iteration,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    NULL,
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    graph_slice->d_row_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    this->edge_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); 


                //util::DisplayDeviceResults(problem->data_slices[0]->d_arank_next,graph_slice->nodes);

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
                        "HITSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "HITSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                if (done[0] == 0) break; 

                NormalizeRank<HITSProblem>(problem, context, 0, graph_slice->nodes);
                NormalizeRank<HITSProblem>(problem, context, 1, graph_slice->nodes); 
                
                iteration++; 

                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,graph_slice->nodes);
                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
                //    graph_slice->nodes); 

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

                if (done[0] == 0 || iteration > max_iteration) break;

                if (DEBUG) printf("\n%lld", (long long) iteration);

            }

            if (retval) break;

        } while(0);

        if (DEBUG) printf("\nGPU HITS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief HITS Enact kernel entry.
     *
     * @tparam HITSProblem HITS Problem type. @see HITSProblem
     *
     * @param[in] problem Pointer to HITSProblem object.
     * @param[in] src Source node for HITS.
     * @param[in] max_grid_size Max grid size for HITS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename HITSProblem>
    cudaError_t Enact(
        CudaContext                          &context,
        HITSProblem                      *problem,
        typename HITSProblem::SizeT       max_iteration,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                HITSProblem,                         // Problem data type
            300,                                // CUDA_ARCH
            INSTRUMENT,                         // INSTRUMENT
            0,                                  // SATURATION QUIT
            true,                               // DEQUEUE_HITSOBLEM_SIZE
            8,                                  // MIN_CTA_OCCUPANCY
            6,                                  // LOG_THREADS
            1,                                  // LOG_LOAD_VEC_SIZE
            0,                                  // LOG_LOADS_PER_TILE
            5,                                  // LOG_RAKING_THREADS
            5,                                  // END_BITMASK_CULL
            8>                                  // LOG_SCHEDULE_GRANULARITY
                VertexMapPolicy;

            typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                HITSProblem,                         // Problem data type
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

            return EnactHITS<EdgeMapPolicy, VertexMapPolicy, HITSProblem>(
                    context, problem, max_iteration, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace pr
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
