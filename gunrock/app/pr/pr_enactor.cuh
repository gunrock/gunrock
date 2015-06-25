// ---------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ---------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------------------

/**
 * @file
 * pr_enactor.cuh
 *
 * @brief PR Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/app/pr/pr_functor.cuh>
#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief PR problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA
 *         clock-count statistics
 */
template<bool INSTRUMENT>
class PREnactor : public EnactorBase
{
protected:

    volatile int *done;
    int          *d_done;
    cudaEvent_t  throttle_event;

    /**
     * @brief Prepare the enactor for PR kernel call.
     *        Must be called prior to each PR search.
     *
     * @param[in] problem PR Problem object which holds the
     *            graph data and PR problem data to compute.
     *
     * \return cudaError_t object which indicates the success of all CUDA calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(ProblemData *problem) {
        typedef typename ProblemData::SizeT    SizeT;
        typedef typename ProblemData::VertexId VertexId;
        cudaError_t retval = cudaSuccess;
        do {
            //initialize the host-mapped "done"
            if (!done) {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::GRError(
                        cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                        "PREnactor cudaHostAlloc done failed",
                        __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(
                        cudaHostGetDevicePointer((void**)&d_done,(void*)done,0),
                        "PREnactor cudaHostGetDevicePointer done failed",
                        __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(
                        cudaEventCreateWithFlags(
                            &throttle_event, cudaEventDisableTiming),
                        "PREnactor createWithFlags throttle_event failed",
                        __FILE__, __LINE__)) break;
            }

            done[0] = -1;

            //graph slice
            typename ProblemData::GraphSlice
                *graph_slice = problem->graph_slices[0];

            // Bind row-offsets texture
            cudaChannelFormatDesc
                row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc
                = row_offsets_desc;
            if (retval = util::GRError(
                    cudaBindTexture(
                        0,
                        oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                        graph_slice->d_row_offsets,
                        (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "PREnactor cudaBindTexture row_offset_tex_ref failed",
                    __FILE__, __LINE__)) break;

            /*
              cudaChannelFormatDesc
              column_indices_desc = cudaCreateChannelDesc<VertexId>();
              oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc
              = column_indicies_desc;
              if (retval = util::GRError(cudaBindTexture(
              0,
              gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
              graph_slice->d_column_indices,
              graph_slice->edges * sizeof(VertexId)),
              "PREnactor cudaBindTexture column_indices_tex_ref failed",
              __FILE__, __LINE__)) break;
            */
        } while (0);

        return retval;
    }

public:

    /**
     * @brief PREnactor constructor
     */
    PREnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG), done(NULL), d_done(NULL) {}

    /**
     * @brief PREnactor destructor
     */
    virtual ~PREnactor() {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                          "PREnactor cudaFreeHost done failed",
                          __FILE__, __LINE__);
            util::GRError(cudaEventDestroy(throttle_event),
                          "PREnactor cudaEventDestroy throttle_event failed",
                          __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last PR search enacted.
     *
     * @param[out] total_queued Total queued elements in PR kernel running.
     * @param[out] avg_duty Average kernel running duty
     *             (kernel run time / kernel lifetime).
     */
    void GetStatistics(
        long long &total_queued,
        double    &avg_duty,
        long long &num_iter) {
        cudaThreadSynchronize();
        total_queued = enactor_stats.total_queued;
        avg_duty = (enactor_stats.total_lifetimes > 0) ?
            double (enactor_stats.total_runtimes) /
            enactor_stats.total_lifetimes : 0.0;
        num_iter = enactor_stats.iteration;
    }

    /** @} */

    /**
     * @brief Enacts a page rank computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam PRProblem PR Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs.
     * @param[in] problem PRProblem object.
     * @param[in] max_iteration Maximum iteration number for PR.
     * @param[in] max_grid_size Max grid size for PR kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename PRProblem>
    cudaError_t EnactPR(
        CudaContext               &context,
        PRProblem                 *problem,
        typename PRProblem::SizeT max_iteration,
        int                       max_grid_size = 0) {
        typedef typename PRProblem::SizeT    SizeT;
        typedef typename PRProblem::VertexId VertexId;
        typedef typename PRProblem::Value    Value;
        typedef PRFunctor<VertexId, SizeT, Value, PRProblem> PrFunctor;
        cudaError_t retval = cudaSuccess;

        do {
            unsigned int *d_scanned_edges = NULL;
            if (DEBUG) {
                printf("Iteration, Advance queue, Filter queue\n0");
            }

            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase::Setup(
                    max_grid_size,
                    AdvanceKernelPolicy::CTA_OCCUPANCY,
                    FilterKernelPolicy::CTA_OCCUPANCY)) break;

            // Single-gpu graph slice
            typename PRProblem::GraphSlice
                *graph_slice = problem->graph_slices[0];
            typename PRProblem::DataSlice
                *data_slice = problem->d_data_slices[0];

            if (AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::LB) {
                if (retval = util::GRError(
                        cudaMalloc(
                            (void**)&d_scanned_edges,
                            graph_slice->edges * sizeof(unsigned int)),
                        "PRProblem cudaMalloc d_scanned_edges failed",
                        __FILE__, __LINE__)) return retval;
            }

            frontier_attribute.queue_length = graph_slice->nodes;
            frontier_attribute.queue_index  = 0; // Work queue index
            frontier_attribute.selector     = 0;
            frontier_attribute.queue_reset  = true;

            cudaEvent_t start, stop;
            float actual_elapsed = 0.0f;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            SizeT frontier_attribute_queue_length = graph_slice->nodes;

            // Step through PageRank iterations
            while (done[0] < 0) {
                // advance kernel
                oprtr::advance::LaunchKernel<
                    AdvanceKernelPolicy, PRProblem, PrFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);

                if (DEBUG && (retval = util::GRError(
                                  cudaThreadSynchronize(),
                                  "advance::Kernel failed",
                                  __FILE__, __LINE__))) break;
                // give host memory mapped visibility to GPU updates
                cudaEventQuery(throttle_event);

                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(
                            frontier_attribute.queue_index+1,
                            frontier_attribute_queue_length)) break;
                    printf(", %lld",
                           (long long) frontier_attribute_queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats.advance_kernel_stats.Accumulate(
                            enactor_stats.advance_grid_size,
                            enactor_stats.total_runtimes,
                            enactor_stats.total_lifetimes)) break;
                }

                // Throttle
                if (enactor_stats.iteration & 1) {
                    if (retval = util::GRError(
                            cudaEventRecord(throttle_event),
                            "PREnactor cudaEventRecord throttle_event failed",
                            __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(
                            cudaEventSynchronize(throttle_event),
                            "PREnactor cudaSynchronize throttle_event failed",
                            __FILE__, __LINE__)) break;
                }

                if (done[0] == 0) break;

                // filter kernel
                oprtr::filter::Kernel<FilterKernelPolicy, PRProblem, PrFunctor>
                    <<<enactor_stats.filter_grid_size,
                    FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                    NULL,
                    NULL,
                    data_slice,
                    NULL,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    enactor_stats.filter_kernel_stats);

                if (DEBUG && (retval = util::GRError(
                                  cudaThreadSynchronize(),
                                  "filter::Kernel failed",
                                  __FILE__, __LINE__))) break;
                // give host memory mapped visibility to GPU updates
                cudaEventQuery(throttle_event);

                ++enactor_stats.iteration;
                ++frontier_attribute.queue_index;

                if (retval = work_progress.GetQueueLength(
                        frontier_attribute.queue_index,
                        frontier_attribute_queue_length)) break;

                // swap rank_curr and rank_next
                util::MemsetCopyVectorKernel<<<128, 128>>>(
                    problem->data_slices[0]->d_rank_curr,
                    problem->data_slices[0]->d_rank_next,
                    graph_slice->nodes);

                util::MemsetKernel<<<128, 128>>>(
                    problem->data_slices[0]->d_rank_next,
                    (Value)0.0, graph_slice->nodes);

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(
                            frontier_attribute.queue_index,
                            frontier_attribute_queue_length)) break;
                    enactor_stats.total_queued +=
                        frontier_attribute_queue_length;
                    if (DEBUG) {
                        printf(", %d", frontier_attribute_queue_length);
                    }
                    if (INSTRUMENT) {
                        if (retval=enactor_stats.filter_kernel_stats.Accumulate(
                                enactor_stats.filter_grid_size,
                                enactor_stats.total_runtimes,
                                enactor_stats.total_lifetimes)) break;
                    }
                }

                if (done[0] == 0 || frontier_attribute_queue_length == 0 ||
                    enactor_stats.iteration >= max_iteration) break;

                if (DEBUG) {
                    printf("\n%lld", (long long) enactor_stats.iteration);
                }
            }

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&actual_elapsed, start, stop);

            printf("PageRank actual_kernel_elapsed: %.4f ms\n", actual_elapsed);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            if (retval) break;

            // sort according to the rank values
            MergesortPairs(problem->data_slices[0]->d_rank_curr,
                           problem->data_slices[0]->d_node_ids,
                           graph_slice->nodes, mgpu::greater<Value>(), context);

            if (d_scanned_edges) cudaFree(d_scanned_edges);

        } while(0);

        if (DEBUG) printf("\nGPU PageRank Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief PR Enact kernel entry.
     *
     * @tparam PRProblem PR Problem type. @see PRProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs.
     * @param[in] problem Pointer to PRProblem object.
     * @param[in] max_iteration Max iterations for PR.
     * @param[in] max_grid_size Max grid size for PR kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA calls.
     */
    template <typename PRProblem>
    cudaError_t Enact(
        CudaContext               &context,
        PRProblem                 *problem,
        typename PRProblem::SizeT max_iteration,
        int                       traversal_mode,
        int                       max_grid_size = 0) {
        if (this->cuda_props.device_sm_version >= 300) {

            typedef gunrock::oprtr::filter::KernelPolicy<
                PRProblem,                          // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                1,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                PRProblem,                          // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                1,                                  // MIN_CTA_OCCUPANCY
                10,                                 // LOG_THREADS
                8,                                  // LOG_LOAD_VEC_SIZE
                32*128,                             // LOG_LOADS_PER_TILE
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                LBAdvanceKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                PRProblem,                          // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                10,                                 // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD
                1,                                  // LOG_LOAD_VEC_SIZE
                1,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::TWC_FORWARD>
                FWDAdvanceKernelPolicy;

            if (traversal_mode == 1) {
                return EnactPR<
                    FWDAdvanceKernelPolicy, FilterKernelPolicy, PRProblem>(
                        context, problem, max_iteration, max_grid_size);
            } else {
                return EnactPR<
                    LBAdvanceKernelPolicy, FilterKernelPolicy, PRProblem>(
                        context, problem, max_iteration, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all architectures

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
