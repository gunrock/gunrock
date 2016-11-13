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
template <typename _Problem>
class SALSAEnactor : public EnactorBase <typename _Problem::SizeT>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    Problem    *problem;
    ContextPtr *context;

    public:

    /**
     * @brief SALSAEnactor constructor
     */
    SALSAEnactor(
        int   num_gpus   = 1, 
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(EDGE_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem(NULL),
        context(NULL)
    {
    }

    /**
     * @brief SALSAEnactor destructor
     */
    virtual ~SALSAEnactor()
    {
    }

    void NormalizeRank(
        //Problem *problem, 
        //CudaContext &context, 
        int hub_or_auth,
        cudaStream_t stream = 0) 
    {
        Value *rank_curr;
        Value *rank_next;
        if (hub_or_auth == 0) {
            rank_curr = problem->data_slices[0]->hrank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->hrank_next.GetPointer(util::DEVICE);
            //printf("hub\n");
        } else {
            rank_curr = problem->data_slices[0]->arank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->arank_next.GetPointer(util::DEVICE);
            //printf("auth\n");
        }

        //swap rank_curr and rank_next
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            rank_curr, rank_next, this -> problem -> nodes);

        util::MemsetKernel<<<128, 128, 0, stream>>>(
            rank_next, (Value)0.0, this -> problem -> nodes);

        //util::DisplayDeviceResults(rank_curr, nodes);
    }

    template<
        typename AdvancekernelPolicy,
        typename FilterkernelPolicy>
    cudaError_t InitSALSA(
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
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */

    /**
     * @brief Enacts a SALSA computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance
     * @tparam FilterKernelPolicy Kernel policy for filter
     *
     * @param[in] max_iteration Max number of iterations of SALSA algorithm
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactSALSA(
        SizeT        max_iteration)
    {
        typedef HFORWARDFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> HForwardFunctor;

        typedef AFORWARDFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> AForwardFunctor;

        typedef HBACKWARDFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> HBackwardFunctor;

        typedef ABACKWARDFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> ABackwardFunctor;

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
        SizeT        *d_scanned_edges    = NULL;
        SizeT         nodes              = graph_slice -> nodes;
        SizeT         edges              = graph_slice -> edges;

        if (this -> debug) 
        {
            printf("Iteration, Edge map queue, Vertex map queue\n");
            printf("0");
            fflush(stdout);
        }

        //cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();

        frontier_attribute->queue_length     = nodes;
        frontier_attribute->queue_index      = 0;
        frontier_attribute->selector         = 0;
        frontier_attribute->queue_reset      = true;

        if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) 
        {
            //if (retval = util::GRError(cudaMalloc(
            //    (void**)&d_scanned_edges,
            //    graph_slice->edges * sizeof(SizeT)),
            //    "SALSAProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) 
            //    return retval;
            if (retval = data_slice -> scanned_edges[0].EnsureSize(edges))
                return retval;
            d_scanned_edges = data_slice -> scanned_edges[0].GetPointer(util::DEVICE);
        }

        // Step through SALSA iterations
        {

            util::MemsetIdxKernel<<<128, 128, 0, stream>>>(
                frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                nodes);

            //if (retval = work_progress->SetQueueLength(
            //    frontier_attribute -> queue_index, 
            //    nodes)) 
            //    break;

            // Edge Map
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, HForwardFunctor, gunrock::oprtr::advance::V2E>(
                enactor_stats[0],
                frontier_attribute[0],
                typename HForwardFunctor::LabelT(),
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), //d_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), //d_out_queue
                (Value*   )NULL,    //d_pred_in_queue
                (Value*   )NULL,
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,
                graph_slice->edges,
                work_progress[0],
                context[0],
                stream);

            if (this -> debug)
            {
               if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__)) 
                    return retval;
            }

            // Edge Map
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, AForwardFunctor, gunrock::oprtr::advance::V2E>(
                enactor_stats[0],
                frontier_attribute[0],
                typename AForwardFunctor::LabelT(),
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), //d_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), //d_out_queue
                (Value*   )NULL,    //d_pred_in_queue
                (Value*   )NULL,
                graph_slice->column_offsets.GetPointer(util::DEVICE),
                graph_slice->row_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,
                graph_slice->edges,
                work_progress[0],
                context[0],
                stream);

            if (this -> debug)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__)) 
                    return retval;
            }
        }

        while (true) 
        {
            util::MemsetIdxKernel<<<128, 128, 0, stream>>>(
                frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), 
                edges);

            frontier_attribute->queue_length     = graph_slice->edges;

            // Edge Map
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, HBackwardFunctor, gunrock::oprtr::advance::E2V>(
                enactor_stats[0],
                frontier_attribute[0],
                typename HBackwardFunctor::LabelT(),
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                (Value*   )NULL,
                (Value*   )NULL,
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                graph_slice->column_offsets.GetPointer(util::DEVICE),
                graph_slice->row_indices.GetPointer(util::DEVICE),
                graph_slice->edges,                    // max_in_queue
                graph_slice->edges * 10000,                  // max_out_queue
                work_progress[0],
                context[0],
                stream,
                false,
                true);

            if (this -> debug)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__)) 
                    return retval;
            }

            NormalizeRank(0, stream);

            // Edge Map
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, ABackwardFunctor, gunrock::oprtr::advance::E2V>(
                enactor_stats[0],
                frontier_attribute[0],
                typename ABackwardFunctor::LabelT(),
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                (Value*   )NULL,
                (Value*   )NULL,
                graph_slice->column_offsets.GetPointer(util::DEVICE),
                graph_slice->row_indices.GetPointer(util::DEVICE),
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                graph_slice->edges,                   // max_in_queue
                graph_slice->edges * 10000,                 // max_out_queue
                work_progress[0],
                context[0],
                stream,
                false,
                true);

            if (this -> debug)
            {
                if (retval = work_progress->GetQueueLength(
                    frontier_attribute -> queue_index, 
                    frontier_attribute -> queue_length,
                    false, stream)) 
                    return retval;
                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__))
                    return retval;

                printf(", %lld", (long long)frontier_attribute->queue_length);
            }

            NormalizeRank(1, stream);

            enactor_stats->iteration++;

            if (enactor_stats->iteration >= max_iteration) break;

            if (this -> debug) 
                printf("\n%lld", (long long) enactor_stats->iteration);
        }

        //Check overflow ignored here

        if (this -> debug) printf("\nGPU SALSA Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
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
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32 * 128,                           // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
            AdvanceKernelPolicy;

    /**
     * \addtogroup PublicInterface
     * @{
     */

    cudaError_t Reset()
    {
        return BaseEnactor::Reset();
    }

    /**
     * @brief SALSA Enact kernel entry.
     *
     * @tparam SALSAProblem SALSA Problem type. @see SALSAProblem
     *
     * @param[in] context CUDA Contet pointer.
     * @param[in] problem Pointer to SALSAProblem object.
     * @param[in] max_grid_size Max grid size for SALSA kernel calls.
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
            return InitSALSA<AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief SALSA Enact kernel entry.
     *
     * @tparam SALSAProblem SALSA Problem type. @see SALSAProblem
     *
     * @param[in] max_iteration Max number of iterations.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        SizeT         max_iteration)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) 
        {
            return EnactSALSA<AdvanceKernelPolicy, FilterKernelPolicy>(
                    max_iteration);
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
