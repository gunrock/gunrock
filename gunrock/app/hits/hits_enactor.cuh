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

#include <thread>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>
#include <gunrock/util/sharedmem.cuh>
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
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
template <typename _Problem>
class HITSEnactor : public EnactorBase<typename _Problem::SizeT>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    Problem    *problem;
    ContextPtr *context;

    // Members
    protected:


    /**
     * Current iteration, also used to get the final search depth of the HITS search
     */
    long long                           iteration;

    // Methods
    protected:

    public:

    /**
     * @brief HITSEnactor constructor
     */
    HITSEnactor(
        int    num_gpus   = 1, 
        int   *gpu_idx    = NULL,
        bool   instrument = false,
        bool   debug      = false,
        bool   size_check = true) :
        BaseEnactor(
            EDGE_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem (NULL),
        context (NULL),
        iteration (0)
    {
    }

    /**
     * @brief HITSEnactor destructor
     */
    virtual ~HITSEnactor()
    {
    }

    void NormalizeRank(int hub_or_auth, cudaStream_t stream = 0)
    {

        Value *rank_curr;
        Value *rank_next;
        if (hub_or_auth == 0) {
            rank_curr = problem->data_slices[0]->hrank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->hrank_next.GetPointer(util::DEVICE);
        } else {
            rank_curr = problem->data_slices[0]->arank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->arank_next.GetPointer(util::DEVICE);
        }

        //swap rank_curr and rank_next
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            rank_curr, rank_next, this -> problem -> nodes);

        util::MemsetKernel<<<128, 128>>>(
            rank_next, (Value)0.0, this -> problem -> nodes);

        //util::DisplayDeviceResults(rank_curr, nodes);
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */

    template<
        typename AdvancekernelPolicy,
        typename FilterkernelPolicy>
    cudaError_t InitHITS(
        ContextPtr *context,
        Problem    *problem,
        int         max_grid_size = 0)
    {   
        cudaError_t retval = cudaSuccess;

        if (retval = BaseEnactor::Init(
            //problem,
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) 
            return retval;

        this -> problem = problem;
        this -> context = context;
        return retval;
    }   

    /**
     * @brief Enacts a HITS computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance.
     * @tparam FilterKernelPolicy Kernel policy for filter.
     *
     * @param[in] max_iteration Max number of iterations of HITS algorithm
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactHITS(
        SizeT            max_iteration)
    {
        typedef HUBFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> HubFunctor;

        typedef AUTHFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> AuthFunctor;

        GraphSlice<VertexId, SizeT, Value>
                     *graph_slice        = problem->graph_slices       [0];
        FrontierAttribute<SizeT>
                     *frontier_attribute = &this->frontier_attribute   [0];
        EnactorStats<SizeT> *enactor_stats      = &this->enactor_stats        [0];
        // Single-gpu graph slice
        typename Problem::DataSlice
                     *data_slice         =  problem->data_slices       [0];
        typename Problem::DataSlice
                     *d_data_slice       =  problem->d_data_slices     [0];
        util::DoubleBuffer<VertexId, SizeT, Value>
                     *frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime<SizeT>
                     *work_progress      = &this->work_progress        [0];
        cudaStream_t  stream             =  data_slice->streams        [0];
        ContextPtr    context            =  this -> context            [0];
        cudaError_t   retval             = cudaSuccess;

        do {
            if (this -> debug) 
            {
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
                fflush(stdout);
            }

            // Single-gpu graph slice
            //typename HITSProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            //typename HITSProblem::DataSlice *data_slice = problem->d_data_slices[0];

            // Bind row-offsets texture
            //cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();

            frontier_attribute->queue_length         = graph_slice->nodes;
            frontier_attribute->queue_index          = 0;        // Work queue index
            frontier_attribute->selector             = 0;
            frontier_attribute->queue_reset          = true;

            // Step through HITS iterations
            while (true) 
            {
                //if (retval = work_progress -> SetQueueLength(
                //    frontier_attribute -> queue_index, 
                //    frontier_attribute -> queue_length)) break;
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, AuthFunctor, gunrock::oprtr::advance::V2V>(
                    enactor_stats[0],
                    frontier_attribute[0],
                    typename AuthFunctor::LabelT(),
                    data_slice,
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    (SizeT*   )NULL,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    (Value*   )NULL,
                    (Value*   )NULL,
                    graph_slice->column_offsets.GetPointer(util::DEVICE),
                    graph_slice->row_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    work_progress[0],
                    context[0],
                    stream);

                if (this -> debug)
                {
                    if (retval = util::GRError(cudaStreamSynchronize(stream), 
                        "edge_map_forward::Kernel failed", __FILE__, __LINE__)) 
                        break;
                }

                //util::DisplayDeviceResults(problem->data_slices[0]->d_arank_next,graph_slice->nodes);
                NormalizeRank(1, stream);

                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel
                    <AdvanceKernelPolicy, Problem, HubFunctor, gunrock::oprtr::advance::V2V>(
                    enactor_stats[0],
                    frontier_attribute[0],
                    typename HubFunctor::LabelT(),
                    data_slice,
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    (SizeT*   )NULL,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    (Value*   )NULL,
                    (Value*   )NULL,
                    graph_slice->row_offsets.GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    work_progress[0],
                    context[0],
                    stream);

                //util::DisplayDeviceResults(problem->data_slices[0]->d_arank_next,graph_slice->nodes);

                if (this -> debug)
                {
                    if (retval = work_progress -> GetQueueLength(
                        frontier_attribute -> queue_index, 
                        frontier_attribute -> queue_length,
                        false, stream)) break;

                    if (retval = util::GRError(cudaStreamSynchronize(stream), 
                        "edge_map_forward::Kernel failed", __FILE__, __LINE__)) break;

                    printf(", %lld", (long long)frontier_attribute->queue_length);
                }

                //if (this -> instrument) {
                //    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                //        enactor_stats->advance_grid_size,
                //        enactor_stats->total_runtimes,
                //        enactor_stats->total_lifetimes)) break;
                //}

                NormalizeRank(0, stream);

                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,graph_slice->nodes);
                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
                //    graph_slice->nodes);

                enactor_stats->iteration++;

                if (enactor_stats->iteration >= max_iteration) break;

                if (this -> debug) 
                    printf("\n%lld", (long long) enactor_stats->iteration);

            }

            if (retval) break;

        } while(0);

        if (this -> debug) printf("\nGPU HITS Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                         // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_HITSOBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        6,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
        FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                         // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        6,                                  // LOG_THREADS
        0,                                  // LOG_BLOCKS
        0,                                  // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                            // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
            AdvanceKernelPolicy;


    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief HITS Enact kernel entry.
     *
     * @tparam HITSProblem HITS Problem type. @see HITSProblem
     *
     * @param[in] context CudaContext for moderngpu library
     * @param[in] problem Pointer to HITSProblem object.
     * @param[in] max_grid_size Max grid size for HITS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        ContextPtr *context,
        Problem    *problem,
        int         max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) 
        {
            return InitHITS<AdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief HITS Enact kernel entry.
     *
     * @tparam HITSProblem HITS Problem type. @see HITSProblem
     *
     * @param[in] max_iteration Max iteration number for the algorithm
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        SizeT       max_iteration)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            return EnactHITS<AdvanceKernelPolicy, FilterKernelPolicy>(
                    max_iteration);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace hits
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
