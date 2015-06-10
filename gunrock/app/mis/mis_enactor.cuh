// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mis_enactor.cuh
 *
 * @brief MIS Problem Enactor
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
#include <gunrock/app/mis/mis_problem.cuh>
#include <gunrock/app/mis/mis_functor.cuh>
#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace mis {

/**
 * @brief MIS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class MISEnactor : public EnactorBase
{
    // Members
    protected:

    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for MIS kernel call. Must be called prior to each MIS search.
     * * @param[in] problem MIS Problem object which holds the graph data and MIS problem data to compute.
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
            if (!done) {
                int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "MISEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "MISEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "MISEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            done[0]             = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "MISEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

        } while (0);

        return retval;
    }

    public:

    /**
     * @brief MISEnactor constructor
     */
    MISEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief MISEnactor destructor
     */
    virtual ~MISEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "MISEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "MISEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last MIS search enacted.
     *
     * @param[out] total_queued Total queued elements in MIS kernel running.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = enactor_stats.total_queued;

        avg_duty = (enactor_stats.total_lifetimes >0) ?
            double(enactor_stats.total_runtimes) / enactor_stats.total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a page rank computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam MISProblem MIS Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs.
     * @param[in] problem MISProblem object.
     * @param[in] max_iteration Maximum iteration number for MIS.
     * @param[in] max_grid_size Max grid size for MIS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename MISProblem>
    cudaError_t EnactMIS(
    CudaContext                        &context,
    MISProblem                          *problem,
    typename MISProblem::SizeT           max_iteration,
    int                                 max_grid_size = 0)
    {
        typedef typename MISProblem::SizeT       SizeT;
        typedef typename MISProblem::VertexId    VertexId;
        typedef typename MISProblem::Value       Value;

        typedef MISFunctor<
            VertexId,
            SizeT,
            Value,
            MISProblem> MisFunctor;

        cudaError_t retval = cudaSuccess;

        do {

            unsigned int *d_scanned_edges = NULL;
            if (DEBUG) {
                printf("Iteration, Advance queue, Filter queue\n");
                printf("0");
            }

            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase::Setup(max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY))
                                            break;

            // Single-gpu graph slice
            typename MISProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename MISProblem::DataSlice *data_slice = problem->d_data_slices[0];


            //util::DisplayDeviceResults(problem->data_slices[0]->d_labels, graph_slice->nodes);

            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_scanned_edges,
                                graph_slice->edges * sizeof(unsigned int)),
                            "MISProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            }

            frontier_attribute.queue_length         = graph_slice->nodes;
            frontier_attribute.queue_index          = 0; // Work queue index
            frontier_attribute.selector             = 0;
            frontier_attribute.queue_reset          = true;

            fflush(stdout);

            while (done[0] < 0 && frontier_attribute.queue_length > 0) {

            //Advance with GatherReduce
            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, MISProblem, MisFunctor>(
                d_done,
                enactor_stats,
                frontier_attribute,
                data_slice,
                (VertexId*)NULL,
                (bool*)NULL,
                (bool*)NULL,
                d_scanned_edges,
                graph_slice->frontier_queues.d_keys[frontier_attribute.selector],   //d_in_queue
                graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1], //d_out_queue
                (VertexId*)NULL,    // d_pred_in_queue
                graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],
                graph_slice->d_row_offsets,
                graph_slice->d_column_indices,
                (SizeT*)NULL,
                (VertexId*)NULL,
                graph_slice->nodes,
                graph_slice->edges,
                this->work_progress,
                context,
                gunrock::oprtr::advance::V2V,
                false, //not inverse_graph
                gunrock::oprtr::advance::MAXIMUM,   //REDUCE_OP
                gunrock::oprtr::advance::VERTEX,    //REDUCE_TYPE (get reduced value from a |V| array
                problem->data_slices[enactor_stats.gpu_id]->d_labels,
                problem->data_slices[enactor_stats.gpu_id]->d_values_to_reduce,
                problem->data_slices[enactor_stats.gpu_id]->d_reduced_values);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "advance::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);

                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    printf(", %lld", (long long) frontier_attribute.queue_length);
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
                        "MISEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "MISEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                //util::DisplayDeviceResults(problem->data_slices[0]->d_reduced_values, frontier_attribute.queue_length);
                //util::DisplayDeviceResults(problem->data_slices[0]->d_values_to_reduce, graph_slice->edges);

                //printf("queuelength before filter%d\n", frontier_attribute.queue_length);
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);


                //Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, MISProblem, MisFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration+1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
                    NULL,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
                    data_slice,
                    NULL,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats.filter_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    //printf("queuelength after filter%d\n", frontier_attribute.queue_length);

                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                    }

                //util::DisplayDeviceResults(problem->data_slices[0]->d_mis_ids, graph_slice->nodes);

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    enactor_stats.total_queued += frontier_attribute.queue_length;
                    if (DEBUG) printf(", %lld", (long long) frontier_attribute.queue_length);
                    if (INSTRUMENT) {
                        if (retval = enactor_stats.filter_kernel_stats.Accumulate(
                            enactor_stats.filter_grid_size,
                            enactor_stats.total_runtimes,
                            enactor_stats.total_lifetimes)) break;
                    }
                }
                // Check if done
                if (done[0] == 0) break;

                enactor_stats.iteration++;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }
            
            if (retval) break;

            if (d_scanned_edges) cudaFree(d_scanned_edges);

        } while(0);

        if (DEBUG) printf("\nGPU Maximal Independent Set Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief MIS Enact kernel entry.
     *
     * @tparam MISProblem MIS Problem type. @see MISProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs.
     * @param[in] problem Pointer to MISProblem object.
     * @param[in] max_iteration Max iterations for MIS.
     * @param[in] max_grid_size Max grid size for MIS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename MISProblem>
    cudaError_t Enact(
        CudaContext               &context,
        MISProblem                 *problem,
        typename MISProblem::SizeT max_iteration,
        int                       max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                MISProblem,                          // Problem data type
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
                MISProblem,                          // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                1,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                AdvanceKernelPolicy;


            return EnactMIS<AdvanceKernelPolicy, FilterKernelPolicy, MISProblem>(
                context, problem, max_iteration, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace mis
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
