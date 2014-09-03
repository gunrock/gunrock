// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
#include <gunrock/priority_queue/kernel.cuh>
#include <gunrock/priority_queue/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>


namespace gunrock {
namespace app {
namespace sssp {

/**
 * @brief SSSP problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class SSSPEnactor : public EnactorBase
{
    // Members
    protected:

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
     * Current iteration, also used to get the final search depth of the SSSP search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for SSSP kernel call. Must be called prior to each SSSP search.
     *
     * @param[in] problem SSSP Problem object which holds the graph data and SSSP problem data to compute.
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
                    "SSSPEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "SSSPEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "SSSPEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            done[0]             = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets and bitmask texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "SSSPEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            graph_slice->edges * sizeof(VertexId)),
                        "SSSPEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief SSSPEnactor constructor
     */
    SSSPEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief SSSPEnactor destructor
     */
    virtual ~SSSPEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "SSSPEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "SSSPEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last SSSP search enacted.
     *
     * @param[out] total_queued Total queued elements in SSSP kernel running.
     * @param[out] search_depth Search depth of SSSP algorithm.
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
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam SSSPProblem SSSP Problem type.
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename SSSPProblem>
    cudaError_t EnactSSSP(
    CudaContext                         &context,
    SSSPProblem                          *problem,
    typename SSSPProblem::VertexId       src,
    double                              queue_sizing,
    int                                 max_grid_size = 0)
    {
        typedef typename SSSPProblem::SizeT      SizeT;
        typedef typename SSSPProblem::VertexId   VertexId;

        typedef SSSPFunctor<
            VertexId,
            SizeT,
            SSSPProblem> SsspFunctor;

        typedef PQFunctor<
            VertexId,
            SizeT,
            SSSPProblem> PqFunctor;

        typedef gunrock::priority_queue::PriorityQueue<
            VertexId,
            SizeT> NearFarPriorityQueue;

        typedef gunrock::priority_queue::KernelPolicy<
                SSSPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10>                                  // LOG_THREADS
            PriorityQueueKernelPolicy;

        NearFarPriorityQueue *pq = new NearFarPriorityQueue;
        util::GRError(pq->Init(problem->graph_slices[0]->edges, queue_sizing), "Priority Queue SSSP Initialization Failed", __FILE__, __LINE__);

        cudaError_t retval = cudaSuccess;

        unsigned int *d_scanned_edges = NULL;

        do {
            if (DEBUG) {
                printf("Iteration, Advance queue, Filter queue\n");
                printf("0");
            }

            if (retval = Setup(problem)) break;

            // Lazy initialization
            if (retval = EnactorBase::Setup(problem, max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY)) break;

            // Single-gpu graph slice
            typename SSSPProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename SSSPProblem::DataSlice *data_slice = problem->d_data_slices[0];

            fflush(stdout);
            // Step through SSSP iterations

            //for (int iter = 0; iter < graph_slice->nodes; ++iter) {
            
            frontier_attribute.queue_length          = 1;
            frontier_attribute.queue_index        = 0;        // Work queue index
            frontier_attribute.selector                = 0;

            frontier_attribute.queue_reset = true;
            done[0] = -1;

            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_scanned_edges,
                                graph_slice->edges * sizeof(unsigned int)),
                            "SSSPProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            }


            unsigned int pq_level = 0; 
            unsigned int out_length = 0;

            while (out_length > 0 || pq->queue_length > 0 || frontier_attribute.queue_length > 0) {

                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SSSPProblem, SsspFunctor>
                (
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

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "advance::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates

                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                }
    
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    printf(", %lld", (long long) frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], queue_length);
                    //util::DisplayDeviceResults(problem->data_slices[0]->d_labels, graph_slice->nodes);
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
                        "SSSPEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "SSSPEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                // Disable here because of Priority Queue
                //if (done[0] == 0) break;

                // Vertex Map
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, SSSPProblem, SsspFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration+1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
                    NULL,                                                                       // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
                    data_slice,
                    NULL,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats.filter_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;

                //TODO: split the output queue into near/far pile, put far pile in far queue, put near pile as the input queue
                //for next round.
                out_length = 0;
                if (frontier_attribute.queue_length > 0) {
                    out_length = gunrock::priority_queue::Bisect<PriorityQueueKernelPolicy, SSSPProblem, NearFarPriorityQueue, PqFunctor>(
                            (int*)graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                            pq,
                            (unsigned int)frontier_attribute.queue_length,
                            data_slice,
                            graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],
                            pq->queue_length,
                            pq_level,
                            (pq_level+1),
                            context);
                            //printf("out:%d, pq_length:%d\n", out_length, pq->queue_length);
                    if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, out_length)) break;
                }
                //
                //If the output queue is empty and far queue is not, then add priority level and split the far pile.
                if ( out_length == 0 && pq->queue_length > 0) {
                    while (pq->queue_length > 0 && out_length == 0) {
                        pq->selector ^= 1;
                        pq_level++;
                        out_length = gunrock::priority_queue::Bisect<PriorityQueueKernelPolicy, SSSPProblem, NearFarPriorityQueue, PqFunctor>(
                                (int*)pq->nf_pile[0]->d_queue[pq->selector^1],
                                pq,
                                (unsigned int)pq->queue_length,
                                data_slice,
                                graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],
                                0,
                                pq_level,
                                (pq_level+1),
                                context);
                                //printf("out after p:%d, pq_length:%d\n", out_length, pq->queue_length);
                        if (out_length > 0) {
                            if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, out_length)) break;
                        }
                    }
                }

                frontier_attribute.selector ^= 1;
                enactor_stats.iteration++;

                if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;

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
                //if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }
            //}

            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            /*bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }*/
            
        } while(0);
        if (d_scanned_edges) cudaFree(d_scanned_edges);
        delete pq;

        if (DEBUG) printf("\nGPU SSSP Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @tparam SSSPProblem SSSP Problem type. @see SSSPProblem
     *
     * @param[in] context CudaContext pointer for moderngpu APIs
     * @param[in] problem Pointer to SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename SSSPProblem>
    cudaError_t Enact(
        CudaContext                      &context,
        SSSPProblem                      *problem,
        typename SSSPProblem::VertexId    src,
        double                          queue_sizing,
        int                             max_grid_size = 0)
    {
        
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                SSSPProblem,                         // Problem data type
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

            /*typedef gunrock::oprtr::advance::KernelPolicy<
                SSSPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD    
                1,                                  // LOG_LOAD_VEC_SIZE
                1,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::TWC_FORWARD>
                    AdvanceKernelPolicy;*/

            typedef gunrock::oprtr::advance::KernelPolicy<
                SSSPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD    
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                    AdvanceKernelPolicy;

            return EnactSSSP<AdvanceKernelPolicy, FilterKernelPolicy, SSSPProblem>(
                    context, problem, src, queue_sizing, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace sssp
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
