// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_enactor.cuh
 *
 * @brief BC Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>


namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief BC problem enactor class.
 *
 * @tparam INSTRUMENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class BCEnactor : public EnactorBase
{
    // Members
    protected:

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the BC search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for BC kernel call. Must be called prior to each BC search.
     *
     * @param[in] problem BC Problem object which holds the graph data and BC problem data to compute.
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
                    "BCEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "BCEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "BCEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            done[0]             = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets and column_indices texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "BCEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            graph_slice->edges * sizeof(VertexId)),
                        "BCEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief BCEnactor constructor
     */
    BCEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief BCEnactor destructor
     */
    virtual ~BCEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "BCEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "BCEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BC search enacted.
     *
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        double &avg_duty)
    {
        cudaThreadSynchronize();

        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a brandes betweenness centrality computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam BCProblem BC Problem type.
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem BCProblem object.
     * @param[in] src Source node for BC. -1 to compute BC value for each node.
     * @param[in] max_grid_size Max grid size for BC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BCProblem>
    cudaError_t EnactBC(
    CudaContext                         &context,
    BCProblem                          *problem,
    typename BCProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        typedef typename BCProblem::SizeT       SizeT;
        typedef typename BCProblem::VertexId    VertexId;
        typedef typename BCProblem::Value       Value;

        typedef ForwardFunctor<
            VertexId,
            SizeT,
            Value,
            BCProblem> ForwardFunctor;

        typedef BackwardFunctor<
            VertexId,
            SizeT,
            Value,
            BCProblem> BackwardFunctor;

        typedef BackwardFunctor2<
            VertexId,
            SizeT,
            Value,
            BCProblem> BackwardFunctor2;

        cudaError_t retval = cudaSuccess;

        unsigned int    *d_scanned_edges = NULL;
        

        do {
            if (DEBUG) {
                printf("Iteration, Edge map queue, Filter queue\n");
                printf("0");
            }

            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY))
                                            break;

            // Single-gpu graph slice
            typename BCProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename BCProblem::DataSlice *data_slice = problem->d_data_slices[0];

            frontier_attribute.queue_length         = 1;
            frontier_attribute.queue_index          = 0;        // Work queue index
            frontier_attribute.selector             = 0;

            frontier_attribute.queue_reset          = true;

            std::vector<SizeT> forward_queue_offsets(graph_slice->nodes);
            forward_queue_offsets.push_back(0);

            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_scanned_edges,
                                graph_slice->edges * sizeof(unsigned int)),
                            "BCProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            }

            fflush(stdout);

            //util::MemsetAddKernel<<<128, 128>>>(d_scanned_edges, (unsigned int)0, graph_slice->edges);
            // Forward BC iteration
            while (done[0] < 0) {

                if (frontier_attribute.queue_length > 0 && enactor_stats.iteration > 0) {
                    SizeT cur_offset = forward_queue_offsets.back();
                    //printf("offset:%d, current length:%d\n", cur_offset, frontier_attribute.queue_length);
                    util::MemsetCopyVectorKernel<<<128, 128>>>(&problem->data_slices[0]->d_forward_output[cur_offset], graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(&problem->data_slices[0]->d_forward_output[cur_offset], frontier_attribute.queue_length);
                    forward_queue_offsets.push_back(frontier_attribute.queue_length+cur_offset);
                }

                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BCProblem, ForwardFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);

                // Only need to reset queue for once
                if (frontier_attribute.queue_reset)
                    frontier_attribute.queue_reset = false;

                if (/*DEBUG &&*/ (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates

                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    }

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
                        "BCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BCProblem, ForwardFunctor>
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

                if (/*DEBUG &&*/ (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;
                enactor_stats.iteration++;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    } 

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
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

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }
            //delete[] sigmas;
            //delete[] labels;
            //delete[] vids;
            //util::DisplayDeviceResults(problem->data_slices[0]->d_forward_output, forward_queue_offsets.back());

            /*enactor_stats.iteration                 = enactor_stats.iteration - 2;

            frontier_attribute.queue_length        = graph_slice->nodes;
            frontier_attribute.queue_index         = 0;        // Work queue index
            frontier_attribute.selector            = 0;
            frontier_attribute.queue_reset         = true;
            done[0]             = -1;

            // Prepare the label array
            VertexId            label_adjust = -enactor_stats.iteration;
            util::MemsetAddKernel<<<128, 128>>>(problem->data_slices[0]->d_labels, label_adjust, graph_slice->nodes);*/


            if (DEBUG) printf("\nStart backward phase\n%lld", (long long) enactor_stats.iteration);
            // Backward BC iteration
            for (int iter = forward_queue_offsets.size()-2; iter >=0; --iter) {
                frontier_attribute.queue_length = forward_queue_offsets[iter+1]-forward_queue_offsets[iter];  
                /*frontier_attribute.queue_length        = graph_slice->nodes;
                // Fill in the frontier_queues
                util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_keys[0], graph_slice->nodes);

                // Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BCProblem, BackwardFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    -1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[0],      // d_in_queue
                    NULL,
                    graph_slice->frontier_queues.d_keys[1],    // d_out_queue
                    data_slice,
                    NULL,
                    work_progress,
                    graph_slice->nodes,           // max_in_queue
                    graph_slice->edges,         // max_out_queue
                    enactor_stats.filter_kernel_stats);*/


                // Only need to reset queue for once
                /*if (frontier_attribute.queue_reset)
                    frontier_attribute.queue_reset = false; */

                //if (/*DEBUG &&*/ (retval = util::GRError(cudaThreadSynchronize(), "edge_map_backward::Kernel failed", __FILE__, __LINE__))) break;
                /*cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates

                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    }

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
                        "BCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;*/
                // Edge Map
                if (iter > 0) {
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BCProblem, BackwardFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    &problem->data_slices[0]->d_forward_output[forward_queue_offsets[iter]],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[0],            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,                 // max_in_queue
                    graph_slice->edges,                 // max_out_queue
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);
                    } else {
                    gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BCProblem, BackwardFunctor2>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    &problem->data_slices[0]->d_forward_output[forward_queue_offsets[iter]],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[0],            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,                 // max_in_queue
                    graph_slice->edges,                 // max_out_queue
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);
                    }

                if (/*DEBUG &&*/ (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                //frontier_attribute.queue_index++;
                //frontier_attribute.selector ^= 1;

                //util::MemsetAddKernel<<<128, 128>>>(problem->data_slices[0]->d_labels, 1, graph_slice->nodes);

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
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

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration-1);

            }
            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
            
        } while(0);

        if (d_scanned_edges) cudaFree(d_scanned_edges);

        if (DEBUG) printf("\nGPU BC Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BC Enact kernel entry.
     *
     * @tparam BCProblem BC Problem type. @see BCProblem
     *
     * @param[in] context CudaContedt pointer for moderngpu APIs
     * @param[in] problem Pointer to BCProblem object.
     * @param[in] src Source node for BC. -1 indicates computing BC value for all nodes.
     * @param[in] max_grid_size Max grid size for BC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename BCProblem>
    cudaError_t Enact(
        CudaContext                     &context,
        BCProblem                       *problem,
        typename BCProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                BCProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END BIT_MASK (no bitmask cull in BC)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                BCProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::TWC_BACKWARD>
                AdvanceKernelPolicy;

                return EnactBC<AdvanceKernelPolicy, FilterKernelPolicy, BCProblem>(
                context, problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace bc
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
