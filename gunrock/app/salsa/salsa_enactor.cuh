// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * salsa_enactor.cuh
 *
 * @brief SALSA (Stochastic Approach for Link-Structure Analysis) Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/salsa/salsa_problem.cuh>
#include <gunrock/app/salsa/salsa_functor.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace salsa {

/**
 * @brief SALSA problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class SALSAEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime filter_kernel_stats;

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
     * Current iteration, also used to get the final search depth of the SALSA search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for SALSA kernel call. Must be called prior to each SALSA search.
     *
     * @param[in] problem SALSA Problem object which holds the graph data and SALSA problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] vertex_map_grid_size CTA occupancy for vertex mapping kernel call.
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

        //initialize the host-mapped "done"
        if (!done) {
            int flags = cudaHostAllocMapped;

            // Allocate pinned memory for done
            if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                        "SALSAEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) return retval;

            // Map done into GPU space
            if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                        "SALSAEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) return retval;

            // Create throttle event
            if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                        "SALSAEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) return retval;
        }

        done[0]             = -1; 

        return retval;
    }

    public:

    /**
     * @brief SALSAEnactor constructor
     */
    SALSAEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief SALSAEnactor destructor
     */
    virtual ~SALSAEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "SALSAEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "SALSAEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
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

        util::MemsetKernel<<<128, 128>>>(rank_next, (Value)0.0, nodes);

        //util::DisplayDeviceResults(rank_curr, nodes);
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last SALSA search enacted.
     *
     * @param[out] total_queued Total queued elements in SALSA kernel running.
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
     * @brief Enacts a SALSA computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance
     * @tparam FilterKernelPolicy Kernel policy for filter
     * @tparam SALSAProblem SALSA Problem type.
     *
     * @param[in] problem SALSAProblem object.
     * @param[in] max_iteration Max number of iterations of SALSA algorithm
     * @param[in] max_grid_size Max grid size for SALSA kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename SALSAProblem>
    cudaError_t EnactSALSA(
    CudaContext                         &context,
    SALSAProblem                        *problem,
    typename SALSAProblem::SizeT        max_iteration,
    int                                 max_grid_size = 0)
    {
        typedef typename SALSAProblem::SizeT       SizeT;
        typedef typename SALSAProblem::VertexId    VertexId;
        typedef typename SALSAProblem::Value       Value;

        typedef FORWARDFunctor<
            VertexId,
            SizeT,
            Value,
            SALSAProblem> ForwardFunctor;

        typedef BACKWARDFunctor<
            VertexId,
            SizeT,
            Value,
            SALSAProblem> BackwardFunctor;

        cudaError_t retval = cudaSuccess;

        do {
            if (DEBUG) {
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }

            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY))
                                            break;

            //graph slice
            typename SALSAProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename SALSAProblem::DataSlice *data_slice = problem->d_data_slices[0];
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>(); 

            frontier_attribute.queue_length     = graph_slice->nodes;
            frontier_attribute.queue_index      = 0;
            frontier_attribute.selector         = 0;

            frontier_attribute.queue_reset      = true;

            int edge_map_queue_len = frontier_attribute.queue_length;

            // Step through SALSA iterations 
            while (done[0] < 0) {

            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "SALSAEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

                if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, edge_map_queue_len)) break;
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SALSAProblem, ForwardFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    (unsigned int*)NULL,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],               //d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],             //d_out_queue
                    (VertexId*)NULL,    //d_pred_in_queue
                    (VertexId*)NULL,
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2E);
                    

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates

                frontier_attribute.queue_index++;
                frontier_attribute.selector^=1;


                if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);

                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_column_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "SALSAEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SALSAProblem, BackwardFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    (unsigned int*)NULL,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->d_column_offsets,
                    graph_slice->d_row_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::E2V);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); 


                //util::DisplayDeviceResults(problem->data_slices[0]->d_arank_next,graph_slice->nodes);

                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    printf(", %lld", (long long)frontier_attribute.queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats.advance_kernel_stats.Accumulate(
                        enactor_stats.advance_grid_size,
                        enactor_stats.total_runtimes,
                        enactor_stats.total_lifetimes)) break;
                }

                // Throttle
                if (enactor_stats.iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "SALSAEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "SALSAEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                if (done[0] == 0) break; 

                NormalizeRank<SALSAProblem>(problem, context, 0, graph_slice->nodes);
                NormalizeRank<SALSAProblem>(problem, context, 1, graph_slice->nodes); 
                
                enactor_stats.iteration++; 

                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,graph_slice->nodes);
                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
                //    graph_slice->nodes); 

                if (done[0] == 0 || enactor_stats.iteration >= max_iteration) break;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }

            if (retval) break;

            //Check overflow ignored here

        } while(0);

        if (DEBUG) printf("\nGPU SALSA Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SALSA Enact kernel entry.
     *
     * @tparam SALSAProblem SALSA Problem type. @see SALSAProblem
     *
     * @param[in] problem Pointer to SALSAProblem object.
     * @param[in] src Source node for SALSA.
     * @param[in] max_grid_size Max grid size for SALSA kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename SALSAProblem>
    cudaError_t Enact(
        CudaContext                          &context,
        SALSAProblem                        *problem,
        typename SALSAProblem::SizeT       max_iteration,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                SALSAProblem,                         // Problem data type
            300,                                // CUDA_ARCH
            INSTRUMENT,                         // INSTRUMENT
            0,                                  // SATURATION QUIT
            true,                               // DEQUEUE_SALSAOBLEM_SIZE
            8,                                  // MIN_CTA_OCCUPANCY
            6,                                  // LOG_THREADS
            1,                                  // LOG_LOAD_VEC_SIZE
            0,                                  // LOG_LOADS_PER_TILE
            5,                                  // LOG_RAKING_THREADS
            5,                                  // END_BITMASK_CULL
            8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                SALSAProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                0,                                  // LOG_BLOCKS
                0,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::TWC_FORWARD>
                    AdvanceKernelPolicy;

            return EnactSALSA<AdvanceKernelPolicy, FilterKernelPolicy, SALSAProblem>(
                    context, problem, max_iteration, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace salsa
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
