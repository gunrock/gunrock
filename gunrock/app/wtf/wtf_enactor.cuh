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
using namespace gunrock::util;

namespace gunrock {
namespace app {
namespace wtf {

/**
 * @brief WTF problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <typename _Problem/*, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK*/>
class WTFEnactor : public EnactorBase <typename _Problem::SizeT/*, _DEBUG, _SIZE_CHECK*/>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    //static const bool INSTRUMENT = _INSTRUMENT;
    //static const bool DEBUG      = _DEBUG;
    //static const bool SIZE_CHECK = _SIZE_CHECK;
    typedef EnactorBase<SizeT>         BaseEnactor;
    Problem    *problem;
    ContextPtr *context;

    public:

    /**
     * @brief WTFEnactor constructor
     */
    WTFEnactor(
        int   num_gpus   = 1,  
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(
            EDGE_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        problem (NULL),
        context (NULL)
    {
    }

    /**
     * @brief WTFEnactor destructor
     */
    virtual ~WTFEnactor()
    {
    }

    void NormalizeRank(int hub_or_auth, cudaStream_t stream = 0)
    {
        Value *rank_curr;
        Value *rank_next;
        if (hub_or_auth == 0) 
        {
            rank_curr = problem->data_slices[0]->rank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->rank_next.GetPointer(util::DEVICE);
            //printf("hub\n");
        } else {
            rank_curr = problem->data_slices[0]->refscore_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->refscore_next.GetPointer(util::DEVICE);
            //printf("auth\n");
        }

        //swap rank_curr and rank_next
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            rank_curr, rank_next, this -> problem -> nodes);

        util::MemsetKernel<<<128, 128, 0, stream>>>(
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
    cudaError_t InitWTF(
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
     * @brief Enacts a page rank computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     *
     * @param[in] src Source node ID for WTF algorithm
     * @param[in] alpha Parameter to determine iteration number
     * @param[in] max_iteration Max iteration number
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactWTF(
        VertexId       src,
        Value          alpha,
        SizeT          max_iteration)
    {
        typedef PRFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> PrFunctor;

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

        typedef COTFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> CotFunctor;

        GraphSlice<VertexId, SizeT, Value>
                     *graph_slice        = problem->graph_slices       [0];
        FrontierAttribute<SizeT>
                     *frontier_attribute = &this->frontier_attribute   [0];
        EnactorStats<SizeT> *enactor_stats      = &this->enactor_stats        [0];
        // Single-gpu graph slice
        typename Problem::DataSlice
                     *data_slice         =  problem->data_slices       [0].GetPointer(util::HOST);
        typename Problem::DataSlice
                     *d_data_slice       =  problem->data_slices       [0].GetPointer(util::DEVICE);
        util::DoubleBuffer<VertexId, SizeT, Value>
                     *frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime<SizeT>
                     *work_progress      = &this->work_progress        [0];
        cudaStream_t  stream             =  data_slice->streams        [0];
        ContextPtr    context            =  this -> context            [0];
        cudaError_t   retval             = cudaSuccess;
        SizeT        *d_scanned_edges    = NULL;
        SizeT         nodes              = graph_slice -> nodes;
        //SizeT         edges              = graph_slice -> edges;
        GpuTimer      gpu_timer;
        float         elapsed;

        if (this -> debug) 
        {
            printf("Iteration, Edge map queue, Vertex map queue\n");
            printf("0");
            fflush(stdout);
        }

        frontier_attribute->queue_length         = nodes;
        frontier_attribute->queue_index          = 0;        // Work queue index
        frontier_attribute->selector             = 0;
        frontier_attribute->queue_reset          = true;

        SizeT edge_map_queue_len = frontier_attribute->queue_length;

        if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) 
        {
            //if (retval = util::GRError(cudaMalloc(
            //    (void**)&d_scanned_edges,
            //    graph_slice->nodes*10 * sizeof(SizeT)),
            //    "WTFProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) 
            //    return retval;
            if (retval = data_slice -> scanned_edges[0].EnsureSize(nodes * 10))
                return retval;
            d_scanned_edges = data_slice -> scanned_edges[0].GetPointer(util::DEVICE);
        }

        gpu_timer.Start();
        // Step through WTF iterations
        while (frontier_attribute->queue_length > 0)
        {
            //if (retval = work_progress.SetQueueLength(
            //    frontier_attribute.queue_index, 
            //    edge_map_queue_len)) break;
            // Edge Map
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, PrFunctor, gunrock::oprtr::advance::V2V>(
                enactor_stats[0],
                frontier_attribute[0],
                typename PrFunctor::LabelT(),
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
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes, // max_in_queue
                graph_slice->edges, // max_out_queue
                work_progress[0],
                context[0],
                stream,
                false,
                false,
                true);

            if (this -> debug)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__))
                    return retval;
                printf(", %lld", (long long) frontier_attribute->queue_length);
            }

            //frontier_attribute->queue_index++;
            //if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;

            //if (this -> instrument) 
            //{
            //    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
            //        enactor_stats->advance_grid_size,
            //        enactor_stats->total_runtimes,
            //        enactor_stats->total_lifetimes)) break;
            //}


            //if (frontier_attribute.queue_reset)
            //    frontier_attribute.queue_reset = false;

            //if (done[0] == 0) break;
            //if (frontier_attribute->queue_length == 0) break;

            frontier_attribute->queue_length = edge_map_queue_len;

            //if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, edge_map_queue_len)) break;

            // Vertex Map
            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, Problem, PrFunctor>(
                        enactor_stats[0],
                        frontier_attribute[0],
                        (VertexId)0,
                        data_slice,
                        d_data_slice,
                        (SizeT*)NULL,
                        data_slice->visited_mask.GetPointer(util::DEVICE),
                        frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                        frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                        (Value*)NULL,
                        (Value*)NULL, 
                        frontier_attribute->output_length[0],
                        graph_slice->nodes,
                        work_progress[0],
                        context[0],
                        stream,
                        frontier_queue->keys[frontier_attribute->selector  ].GetSize(),
                        frontier_queue->keys[frontier_attribute->selector^1].GetSize(),
                        enactor_stats->filter_kernel_stats,
                        true,
                        false);

            enactor_stats     -> iteration++;
            frontier_attribute-> queue_index++;

            if (retval = work_progress->GetQueueLength(
                frontier_attribute -> queue_index, 
                frontier_attribute -> queue_length,
                false, stream)) 
                return retval;
            if (retval = util::GRError(cudaStreamSynchronize(stream), 
                "filter_forward::Kernel failed", __FILE__, __LINE__)) 
                return retval;

            //num_elements = queue_length;

            //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,
            //    graph_slice->nodes);
            //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
            //    graph_slice->nodes);

            //swap rank_curr and rank_next
            util::MemsetCopyVectorKernel<<<128,128, 0, stream>>>(
                data_slice -> rank_curr.GetPointer(util::DEVICE),
                data_slice -> rank_next.GetPointer(util::DEVICE),
                nodes);
            util::MemsetKernel<<<128, 128, 0, stream>>>(
                data_slice -> rank_next.GetPointer(util::DEVICE),
                (Value)0.0,
                nodes);

            enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
            if (this -> instrument || this -> debug) 
            {
                if (this -> debug) 
                    printf(", %lld", (long long) frontier_attribute->queue_length);
                //if (this -> instrument) 
                //{
                //    if (retval = enactor_stats->filter_kernel_stats.Accumulate(
                //        enactor_stats->filter_grid_size,
                //        enactor_stats->total_runtimes,
                //        enactor_stats->total_lifetimes)) break;
                //}
            }

            if (frontier_attribute->queue_length == 0 || 
                enactor_stats -> iteration > max_iteration) break;

            if (this -> debug) 
                printf("\n%lld", (long long) enactor_stats->iteration);
        }

        gpu_timer.Stop();
        elapsed = gpu_timer.ElapsedMillis();
        printf("PPR Time: %5f\n", elapsed);

        if (retval) return retval;
        enactor_stats->iteration = 0;

        gpu_timer.Start();
        util::CUBRadixSort<Value, VertexId>(
            false, nodes, 
            data_slice -> rank_curr.GetPointer(util::DEVICE), 
            data_slice -> node_ids.GetPointer(util::DEVICE));

        if (retval = util::GRError(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed", __FILE__, __LINE__))
            return retval;

        // 1 according to the first 1000 circle of trust nodes. Get all their neighbors.
        // 2 compute atomicAdd their neighbors' incoming node number.

        frontier_attribute->queue_index          = 0;        // Work queue index
        frontier_attribute->selector             = 0;
        frontier_attribute->queue_reset          = true;
        long long cot_size                       = (1000 > graph_slice->nodes) ? graph_slice->nodes : 1000;
        frontier_attribute->queue_length         = cot_size;

        //if (retval = work_progress.SetQueueLength(frontier_attribute.queue_index, cot_size)) break;

        // Edge Map
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, CotFunctor, gunrock::oprtr::advance::V2V>(
            enactor_stats[0],
            frontier_attribute[0],
            typename CotFunctor::LabelT(),
            data_slice,
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            d_scanned_edges,
            data_slice ->node_ids.GetPointer(util::DEVICE),              // d_in_queue
            frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), // d_out_queue
            (Value*   )NULL,          // d_pred_in_queue
            (Value*   )NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes,
            graph_slice->edges,
            work_progress[0],
            context[0],
            stream,
            false,
            false,
            true);

        gpu_timer.Stop();
        elapsed = gpu_timer.ElapsedMillis();
        printf("CoT Time: %5f\n", elapsed);

        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice -> rank_next    .GetPointer(util::DEVICE), 
            (Value)0.0, nodes);

        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice -> rank_curr    .GetPointer(util::DEVICE), 
            (Value)0.0, nodes);

        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice -> refscore_curr.GetPointer(util::DEVICE), 
            (Value)0.0, nodes);

        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice -> refscore_next.GetPointer(util::DEVICE), 
            (Value)0.0, nodes);

        if (retval = util::GRError(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize failed", __FILE__, __LINE__))
            return retval;

        Value init_score = 1.0;
        if (retval = util::GRError(cudaMemcpy(
            data_slice->rank_curr.GetPointer(util::DEVICE) + src,
            &init_score, sizeof(Value),
            cudaMemcpyHostToDevice),
            "WTFProblem cudaMemcpy d_rank_curr[src] failed", __FILE__, __LINE__)) 
            return retval;

        long long max_salsa_iteration = 1/alpha;
        gpu_timer.Start();

        while (true)
        {
            //if (retval = work_progress.SetQueueLength(
            //    frontier_attribute.queue_index, 
            //    frontier_attribute.queue_length)) break;

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
                d_scanned_edges,
                data_slice->node_ids.GetPointer(util::DEVICE),              // d_in_queue
                frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),            // d_out_queue
                (Value*   )NULL,
                (Value*   )NULL,
                graph_slice->row_offsets.GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes, // max_in_queue
                graph_slice->edges, // max_out_queue
                work_progress[0],
                context[0],
                stream,
                false,
                false,
                true);

            if (this -> debug)
            {
                if (retval = util::GRError(cudaThreadSynchronize(), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__)) 
                    return retval;
            }

            NormalizeRank(1, stream);

            if (retval = work_progress->SetQueueLength(
                frontier_attribute->queue_index, 
                frontier_attribute->queue_length,
                false, stream)) 
                return retval;

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
                d_scanned_edges,
                data_slice->node_ids.GetPointer(util::DEVICE),              // d_in_queue
                frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),            // d_out_queue
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
                stream,
                false,
                false,
                true);

            if (this -> debug)
            {
                if (retval = util::GRError(cudaStreamSynchronize(stream), 
                    "edge_map_forward::Kernel failed", __FILE__, __LINE__)) 
                    return retval;
            }

            NormalizeRank(0, stream);
            enactor_stats->iteration++;
            if (enactor_stats->iteration >= max_salsa_iteration) break;
        }

        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(
            data_slice->node_ids.GetPointer(util::DEVICE), 
            nodes);

        if (retval = util::GRError(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize failed", __FILE__, __LINE__))
            return retval;

        util::CUBRadixSort<Value, VertexId>(
            false, nodes, 
            data_slice->refscore_curr.GetPointer(util::DEVICE), 
            data_slice->node_ids.GetPointer(util::DEVICE));

        gpu_timer.Stop();
        elapsed = gpu_timer.ElapsedMillis();
        printf("WTF Time: %5f\n", elapsed);

        gpu_timer.Stop();
        elapsed = gpu_timer.ElapsedMillis();
        printf("WTF Time: %5f\n", elapsed);

        //if (d_scanned_edges) cudaFree(d_scanned_edges);

        if (this -> debug) printf("\nGPU WTF Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_WTFOBLEM_SIZE
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8>                                  // LOG_SCHEDULE_GRANULARITY
    FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
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

    /**
     * \addtogroup PublicInterface
     * @{
     */

    cudaError_t Reset()
    {
        return BaseEnactor::Reset();
    }

    /**
     * @brief WTF enactor initalization.
     *
     * @tparam WTFProblem WTF Problem type. @see PRProblem
     *
     * @param[in] context CudaContext for moderngpu library
     * @param[in] problem Pointer to WTFProblem object.
     * @param[in] max_grid_size Max grid size for WTF kernel calls.
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
        {
            if (min_sm_version == -1 || 
                this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300) 
        {
            return InitWTF<AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief WTF Enact kernel entry.
     *
     * @tparam WTFProblem WTF Problem type. @see PRProblem
     *
     * @param[in] src Source node for WTF.
     * @param[in] alpha Parameters related to iteration number of WTF algorithm
     * @param[in] max_iteration Max iteration number of WTF algorithm
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        VertexId   src,
        Value      alpha,
        SizeT      max_iteration)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
        {
            if (min_sm_version == -1 || 
                this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300) 
        {
            return EnactWTF<AdvanceKernelPolicy, FilterKernelPolicy>(
                src, alpha, max_iteration);
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
