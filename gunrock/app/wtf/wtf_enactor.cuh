// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * wtf_enactor.cuh
 *
 * @brief WTF Problem Enactor
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
#include <gunrock/app/wtf/wtf_problem.cuh>
#include <gunrock/app/wtf/wtf_functor.cuh>

#include <moderngpu.cuh>

#include <cub/cub.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace wtf {

/**
 * @brief WTF problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class WTFEnactor : public EnactorBase
{
    // Members
    protected:

    volatile int        *done;
    int                 *d_done;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for WTF kernel call. Must be called prior to each WTF search.
     *
     * @param[in] problem WTF Problem object which holds the graph data and WTF problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] filter_grid_size CTA occupancy for vertex mapping kernel call.
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
                    "WTFEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "WTFEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

            }

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
                        "WTFEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief WTFEnactor constructor
     */
    WTFEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * @brief WTFEnactor destructor
     */
    virtual ~WTFEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "WTFEnactor cudaFreeHost done failed", __FILE__, __LINE__);

        }
    }

    template <typename ProblemData>
    void NormalizeRank(ProblemData *problem, CudaContext &context, int hub_or_auth, int nodes)
    {

        typedef typename ProblemData::Value         Value;
        Value *rank_curr;
        Value *rank_next;
        if (hub_or_auth == 0) {
            rank_curr = problem->data_slices[0]->d_rank_curr;
            rank_next = problem->data_slices[0]->d_rank_next;
            //printf("hub\n");
        } else {
            rank_curr = problem->data_slices[0]->d_refscore_curr;
            rank_next = problem->data_slices[0]->d_refscore_next;
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
     * @brief Obtain statistics about the last WTF search enacted.
     *
     * @param[out] total_queued Total queued elements in WTF kernel running.
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
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     * @tparam WTFProblem WTF Problem type.
     *
     * @param[in] problem WTFProblem object.
     * @param[in] src Source node for WTF.
     * @param[in] max_grid_size Max grid size for WTF kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename WTFProblem>
    cudaError_t EnactWTF(
    CudaContext                        &context,
    typename WTFProblem::VertexId       src,
    typename WTFProblem::Value          alpha,
    WTFProblem                          *problem,
    typename WTFProblem::SizeT           max_iteration,
    int                                 max_grid_size = 0)
    {
        typedef typename WTFProblem::SizeT       SizeT;
        typedef typename WTFProblem::VertexId    VertexId;
        typedef typename WTFProblem::Value       Value;

        typedef PRFunctor<
            VertexId,
            SizeT,
            Value,
            WTFProblem> PrFunctor;

        typedef HUBFunctor<
            VertexId,
            SizeT,
            Value,
            WTFProblem> HubFunctor;

        typedef AUTHFunctor<
            VertexId,
            SizeT,
            Value,
            WTFProblem> AuthFunctor;

        typedef COTFunctor<
            VertexId,
            SizeT,
            Value,
            WTFProblem> CotFunctor;

        cudaError_t retval = cudaSuccess;

        unsigned int    *d_scanned_edges = NULL; 
        
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

            // Single-gpu graph slice
            typename WTFProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename WTFProblem::DataSlice *data_slice = problem->d_data_slices[0];

            frontier_attribute.queue_length         = graph_slice->nodes;
            frontier_attribute.queue_index          = 0;        // Work queue index
            frontier_attribute.selector             = 0;

            frontier_attribute.queue_reset          = true;

            int edge_map_queue_len = frontier_attribute.queue_length;

            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (retval = util::GRError(cudaMalloc(
                                (void**)&d_scanned_edges,
                                graph_slice->edges * sizeof(unsigned int)),
                            "PBFSProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            }

            // Step through WTF iterations 
            while (done[0] < 0) {

                //if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, edge_map_queue_len)) break;
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, WTFProblem, PrFunctor>(
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

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

                frontier_attribute.queue_index++;

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


                if (frontier_attribute.queue_reset)
                    frontier_attribute.queue_reset = false;

                if (done[0] == 0) break; 
                
                //if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, edge_map_queue_len)) break;

                // Vertex Map
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, WTFProblem, PrFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration,
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

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;

                enactor_stats.iteration++;
                frontier_attribute.queue_index++;


                if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                
                //num_elements = queue_length;

                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,
                //    graph_slice->nodes);
                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
                //    graph_slice->nodes);
    
                //swap rank_curr and rank_next
                util::MemsetCopyVectorKernel<<<128,
                  128>>>(problem->data_slices[0]->d_rank_curr,
                      problem->data_slices[0]->d_rank_next, graph_slice->nodes);
                util::MemsetKernel<<<128, 128>>>(problem->data_slices[0]->d_rank_next,
                    (Value)0.0, graph_slice->nodes);

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

                if (frontier_attribute.queue_length == 0 || enactor_stats.iteration > max_iteration) break;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }

            if (retval) break;
            enactor_stats.iteration = 0;

        util::CUBRadixSort<Value, VertexId>(false, graph_slice->nodes, problem->data_slices[0]->d_rank_curr, problem->data_slices[0]->d_node_ids);
        
        /*
         * 1 according to the first 1000 circle of trust nodes. Get all their neighbors.
           2 compute atomicAdd their neighbors' incoming node number.
         */
        frontier_attribute.queue_index          = 0;        // Work queue index
        frontier_attribute.selector             = 0;
        frontier_attribute.queue_reset          = true;
        long long cot_size                      = (1000 > graph_slice->nodes) ? graph_slice->nodes : 1000; 
        frontier_attribute.queue_length         = cot_size;

        //if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, cot_size)) break;

        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, WTFProblem, CotFunctor>(
                d_done,
                enactor_stats,
                frontier_attribute,
                data_slice,
                (VertexId*)NULL,
                (bool*)NULL,
                (bool*)NULL,
                d_scanned_edges,
                problem->data_slices[0]->d_node_ids,              // d_in_queue
                graph_slice->frontier_queues.d_keys[frontier_attribute.selector],            // d_out_queue
                (VertexId*)NULL,          // d_pred_in_queue
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

        util::MemsetKernel<<<128, 128>>>(problem->data_slices[0]->d_rank_next, (Value)0.0, graph_slice->nodes);
        util::MemsetKernel<<<128, 128>>>(problem->data_slices[0]->d_rank_curr, (Value)0.0, graph_slice->nodes);

        util::MemsetKernel<<<128, 128>>>(problem->data_slices[0]->d_refscore_curr, (Value)0.0, graph_slice->nodes);
        util::MemsetKernel<<<128, 128>>>(problem->data_slices[0]->d_refscore_next, (Value)0.0, graph_slice->nodes);

        Value init_score = 1.0;
        if (retval = util::GRError(cudaMemcpy(
                        problem->data_slices[0]->d_rank_curr+src,
                        &init_score,
                        sizeof(Value),
                        cudaMemcpyHostToDevice),
                    "WTFProblem cudaMemcpy d_rank_curr[src] failed", __FILE__, __LINE__)) return retval;

        long long max_salsa_iteration = 1/alpha;
        
        while (true) {

            if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;

            // Edge Map
            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, WTFProblem, AuthFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    problem->data_slices[0]->d_node_ids,              // d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],            // d_out_queue
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

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

            NormalizeRank<WTFProblem>(problem, context, 1, graph_slice->nodes);

            if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;

            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, WTFProblem, HubFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    problem->data_slices[0]->d_node_ids,              // d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],            // d_out_queue
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

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

            NormalizeRank<WTFProblem>(problem, context, 0, graph_slice->nodes);

            enactor_stats.iteration++;

            if (enactor_stats.iteration >= max_salsa_iteration) break;
        }

        util::MemsetIdxKernel<<<128, 128>>>(problem->data_slices[0]->d_node_ids, graph_slice->nodes);

        util::CUBRadixSort<Value, VertexId>(false, graph_slice->nodes, problem->data_slices[0]->d_refscore_curr, problem->data_slices[0]->d_node_ids); 

        } while(0); 

        if (d_scanned_edges) cudaFree(d_scanned_edges);

        if (DEBUG) printf("\nGPU WTF Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief WTF Enact kernel entry.
     *
     * @tparam WTFProblem WTF Problem type. @see PRProblem
     *
     * @param[in] problem Pointer to WTFProblem object.
     * @param[in] src Source node for WTF.
     * @param[in] max_grid_size Max grid size for WTF kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename WTFProblem>
    cudaError_t Enact(
        CudaContext                     &context,
        typename WTFProblem::VertexId   src,
        typename WTFProblem::Value      alpha,
        WTFProblem                      *problem,
        typename WTFProblem::SizeT       max_iteration,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                WTFProblem,                         // Problem data type
            300,                                // CUDA_ARCH
            INSTRUMENT,                         // INSTRUMENT
            0,                                  // SATURATION QUIT
            true,                               // DEQUEUE_WTFOBLEM_SIZE
            8,                                  // MIN_CTA_OCCUPANCY
            6,                                  // LOG_THREADS
            1,                                  // LOG_LOAD_VEC_SIZE
            0,                                  // LOG_LOADS_PER_TILE
            5,                                  // LOG_RAKING_THREADS
            5,                                  // END_BITMASK_CULL
            8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                WTFProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,
                32*1024,
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::TWC_FORWARD>
                    AdvanceKernelPolicy;

            return EnactWTF<AdvanceKernelPolicy, FilterKernelPolicy, WTFProblem>(
                    context, src, alpha, problem, max_iteration, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace wtf
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
