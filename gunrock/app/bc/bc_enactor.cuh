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

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
//#include <gunrock/util/scan/multi_scan.cuh>

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

    template <typename Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK> class Enactor;

    template <
        typename VertexId, 
        typename SizeT, 
        typename Value,
        int      NUM_VERTEX_ASSOCIATES, 
        int      NUM_VALUE__ASSOCIATES>
    __global__ void Expand_Incoming_Forward (
        const SizeT            num_elements,
        const VertexId* const  keys_in,
              VertexId*        keys_out,
        const size_t           array_size,
              char*            array)
    {    
        extern __shared__ char s_array[];
        const SizeT STRIDE = gridDim.x * blockDim.x;
        VertexId    t;
        size_t      offset                = 0; 
        VertexId** s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        Value**    s_value__associate_in  = (Value**   )&(s_array[offset]);
        offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
        VertexId** s_vertex_associate_org = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        Value**    s_value__associate_org = (Value**   )&(s_array[offset]);
        SizeT x = threadIdx.x;
        while (x < array_size)
        {    
            s_array[x]=array[x];
            x+=blockDim.x; 
        }    
        __syncthreads();

        x = blockIdx.x * blockDim.x + threadIdx.x;

        while (x<num_elements)
        {    
            VertexId key=keys_in[x];

            if (atomicCAS(s_vertex_associate_org[1]+key, -2, s_vertex_associate_in[1][x])==-2)
            {
                s_vertex_associate_org[0][key]=s_vertex_associate_in[0][x];
                t=-1;
            } else {
                t=atomicCAS(s_vertex_associate_org[0]+key, -1, s_vertex_associate_in[0][x]);
                if (s_vertex_associate_org[0][key]!=s_vertex_associate_in[0][x])
                {
                    keys_out[x]=-1;
                    x+=STRIDE;
                    continue;
                }
            }
            if (t==-1) keys_out[x]=key; else keys_out[x]=-1;
            atomicAdd(s_value__associate_org[0]+key, s_value__associate_in[0][x]);
            x+=STRIDE;
        }
    }

    template <
        typename VertexId, 
        typename SizeT, 
        typename Value,
        int      NUM_VERTEX_ASSOCIATES, 
        int      NUM_VALUE__ASSOCIATES>
    __global__ void Expand_Incoming_Backward (
        const SizeT            num_elements,
        const VertexId* const  keys_in,
              VertexId*        keys_out,
        const size_t           array_size,
              char*            array)
    {    
        extern __shared__ char s_array[];
        const SizeT STRIDE = gridDim.x * blockDim.x;
        size_t      offset                = 0; 
        //VertexId** s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        Value**    s_value__associate_in  = (Value**   )&(s_array[offset]);
        offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
        //VertexId** s_vertex_associate_org = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        Value**    s_value__associate_org = (Value**   )&(s_array[offset]);
        SizeT x = threadIdx.x;
        if (x < array_size)
        {    
            s_array[x]=array[x];
            x+=blockDim.x; 
        }    
        __syncthreads();

        x = blockIdx.x * blockDim.x + threadIdx.x;

        while (x<num_elements)
        {    
            VertexId key=keys_in[x];
            keys_out[x]=key;
            s_value__associate_org[0][key]=s_value__associate_in[0][x];
            s_value__associate_org[1][key]=s_value__associate_in[1][x];
            x+=STRIDE;
        }
    }

template <
    typename AdvanceKernelPolicy, 
    typename FilterKernelPolicy, 
    typename Enactor>
struct Forward_Iteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false, //HAS_SUBQ
    true , //HAS_FULLQ
    false, //BACKWARD
    true , //FORWARD
    true > //UPDATE_PREDECESSORS
{
public:
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    typedef ForwardFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> ForwardFunctor;

    /*static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = false;
    static const bool HAS_FULLQ  = true;
    static const bool BACKWARD   = false;
    static const bool UPDATE_PREDECESSORS = true;*/

    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (enactor_stats->iteration <= 0) return; 

        SizeT cur_offset = data_slice->forward_queue_offsets[peer_].back();
        bool oversized = false;
        //printf("%d\t %lld\t %d\t offset = %d current length = %d resulted_length = %d size = %d\n", thread_num, enactor_stats->iteration, peer_, cur_offset, frontier_attribute->queue_length, cur_offset+frontier_attribute->queue_length, data_slice->forward_output[peer_].GetSize());fflush(stdout);
        if (enactor_stats->retval = 
            Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId> ("forward_output", cur_offset + frontier_attribute->queue_length, &data_slice->forward_output[peer_], oversized, thread_num, enactor_stats->iteration, peer_)) return;
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            data_slice ->forward_output[peer_].GetPointer(util::DEVICE) + cur_offset, 
            frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), 
            frontier_attribute->queue_length);
        //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                //util::DisplayDeviceResults(&problem->data_slices[0]->d_forward_output[cur_offset], frontier_attribute.queue_length);
        data_slice->forward_queue_offsets[peer_].push_back(frontier_attribute->queue_length+cur_offset); 
    }
 
    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        /*if (frontier_attribute->queue_reset && frontier_attribute->queue_length ==0)
        {
            work_progress->SetQueueLength(frontier_attribute->queue_index, 0, false, stream);
            if (Enactor::DEBUG) util::cpu_mt::PrintMessage("return-1", thread_num, enactor_stats->iteration);
            return;
        }*/
        //util::cpu_mt::PrintGPUArray("keys0", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("label0", data_slice->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("sigma0", data_slice->sigmas.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);

        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration);

        frontier_attribute->queue_reset = true;
        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, ForwardFunctor>(
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*)    NULL,
            (bool*)    NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),// d_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
            (VertexId*)NULL,
            (VertexId*)   NULL,
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*)   NULL,
            (VertexId*)NULL,
            graph_slice->nodes, //graph_slice->frontier_elements[frontier_attribute.selector],  // max_in_queue
            graph_slice->edges, //graph_slice->frontier_elements[frontier_attribute.selector^1],// max_out_queue
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);

        frontier_attribute->queue_reset = false;
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        enactor_stats      -> Accumulate(
            work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(frontier_attribute->queue_index), stream);

        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration);
        //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
        //util::cpu_mt::PrintGPUArray("keys1", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("label1", data_slice->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("sigma1", data_slice->sigmas.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);


        if (false) //(DEBUG || INSTRUMENT)
        {
            if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
            //enactor_stats->total_queued += frontier_attribute->queue_length;
            if (Enactor::DEBUG) ShowDebugInfo<Problem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_advance", stream);
            if (Enactor::INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                    enactor_stats->advance_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false,stream)) return;
            }
        }

        // Filter
        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, ForwardFunctor>
            <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats->iteration+1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),// d_in_queue
            NULL,
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
            d_data_slice,
            NULL,
            work_progress[0],
            frontier_queue->keys[frontier_attribute->selector  ].GetSize(),// max_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetSize(),// max_out_queue
            enactor_stats->filter_kernel_stats);

        if (Enactor::DEBUG && (enactor_stats->retval = util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, enactor_stats->iteration);
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream,true)) return;
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //util::cpu_mt::PrintGPUArray("keys3", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("label3", data_slice->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("sigma3", data_slice->sigmas.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);


        if (false) {//(INSTRUMENT || DEBUG) {
            //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
            //enactor_stats->total_queued += frontier_attribute->queue_length;
            if (Enactor::DEBUG) ShowDebugInfo<Problem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_filter", stream);
            if (Enactor::INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                    enactor_stats->filter_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false, stream)) return;
            }
        }
    }

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
              int             grid_size,
              int             block_size,
              size_t          shared_size,
              cudaStream_t    stream,
              SizeT           &num_elements,
        const VertexId* const keys_in,
        util::Array1D<SizeT, VertexId>*       keys_out,
        const size_t          array_size,
              char*           array,
              DataSlice*      data_slice)
    {
        bool over_sized = false;

        Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId>(
            "queue1", num_elements, keys_out, over_sized, -1, -1, -1);
        Expand_Incoming_Forward
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            keys_out->GetPointer(util::DEVICE),
            array_size,
            array);
    }

    static cudaError_t Compute_OutputLength(
        FrontierAttribute<SizeT> *frontier_attribute,
        SizeT       *d_offsets,
        VertexId    *d_indices,
        VertexId    *d_in_key_queue,
        util::Array1D<SizeT, SizeT>       *partitioned_scanned_edges,
        SizeT        max_in,
        SizeT        max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false)
    {
        cudaError_t retval = cudaSuccess;
        bool over_sized = false;
        if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
            "scanned_edges", frontier_attribute->queue_length, partitioned_scanned_edges, over_sized, -1, -1, -1, false)) return retval;
        retval = gunrock::oprtr::advance::ComputeOutputLength
            <AdvanceKernelPolicy, Problem, ForwardFunctor>(
            frontier_attribute,
            d_offsets,
            d_indices,
            d_in_key_queue,
            partitioned_scanned_edges->GetPointer(util::DEVICE),
            max_in,
            max_out,
            context,
            stream,
            ADVANCE_TYPE,
            express); 
        return retval;
    }

    static void Check_Queue_Size(
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        //util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        //DataSlice                     *data_slice,
        //DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice
        //util::CtaWorkProgressLifetime *work_progress,
        //ContextPtr                     context,
        //cudaStream_t                   stream
        )    
    {    
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d, @ %d\n",
                thread_num, iteration, peer_,
                frontier_queue->keys[selector^1].GetSize(),
                request_length, selector);fflush(stdout);

        if (enactor_stats->retval = 
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) return; 
        if (enactor_stats->retval = 
            Check_Size<true, SizeT, VertexId > ("queue3", graph_slice->nodes+2, &frontier_queue->keys  [selector  ], over_sized, thread_num, iteration, peer_, true )) return; 
        if (Problem::USE_DOUBLE_BUFFER)
        {    
            if (enactor_stats->retval = 
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector^1], over_sized, thread_num, iteration, peer_, false)) return; 
            if (enactor_stats->retval = 
                Check_Size<true, SizeT, Value> ("queue3", graph_slice->nodes+2, &frontier_queue->values[selector  ], over_sized, thread_num, iteration, peer_, true )) return; 
        }    
    } 
};

template <typename AdvanceKernelPolicy, typename FilterKernelPolicy, typename Enactor>
struct Backward_Iteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false, //HAS_SUBQ
    true , //HAS_FULLQ
    true , //BACKWARD
    false, //FORWARD
    false> //UPDATE_PREDECESSORS
{
public:
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    typedef BackwardFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> BackwardFunctor;
   typedef BackwardFunctor2<
            VertexId,
            SizeT,
            Value,
            Problem> BackwardFunctor2;
 
    /*static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = false;
    static const bool HAS_FULLQ  = true;
    static const bool BACKWARD   = true;
    static const bool UPDATE_PREDECESSORS = false;*/
   
    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        SizeT cur_pos = data_slice->forward_queue_offsets[peer_].back();
        data_slice->forward_queue_offsets[peer_].pop_back();
        SizeT pre_pos = data_slice->forward_queue_offsets[peer_].back();
        //printf("%d\t %lld\t %d\t offset = %d current length = %d resulted_length = %d size = %d\n", thread_num, enactor_stats->iteration, peer_, pre_pos, cur_pos-pre_pos, cur_pos, data_slice->forward_output[peer_].GetSize());fflush(stdout);
        frontier_attribute->queue_reset  = true;
        frontier_attribute->selector     = 0;//frontier_queue->keys[0].GetSize() > frontier_queue->keys[1].GetSize() ? 1 : 0;
        if (enactor_stats->iteration>0 && cur_pos - pre_pos >0)
        {
            frontier_attribute->queue_length = cur_pos - pre_pos;
            bool over_sized = false;
            if (enactor_stats->retval = Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId> (
                "queue1", frontier_attribute->queue_length, &frontier_queue->keys[frontier_queue->selector], over_sized, thread_num, enactor_stats->iteration, peer_, false)) return;
           util::MemsetCopyVectorKernel<<<256, 256, 0, stream>>>(
                frontier_queue->keys[frontier_queue->selector].GetPointer(util::DEVICE), 
                data_slice ->forward_output[peer_].GetPointer(util::DEVICE) + pre_pos, 
                frontier_attribute->queue_length);
        }
        else frontier_attribute->queue_length = 0;
    }
 
    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
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
        if (enactor_stats->iteration > 0) {
            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, BackwardFunctor>(
                //d_done,
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*)NULL,
                (bool*)NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                NULL, //frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                (VertexId*)NULL,
                (VertexId*)NULL,
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*)NULL,
                (VertexId*)NULL,
                graph_slice->nodes,                 // max_in_queue
                graph_slice->edges,                 // max_out_queue
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false, 
                false);
        } else {
            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, BackwardFunctor2>(
                //d_done,
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*)NULL,
                (bool*)NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                NULL, //frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                (VertexId*)NULL,
                (VertexId*)NULL,
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*)NULL,
                (VertexId*)NULL,
                graph_slice->nodes,                 // max_in_queue
                graph_slice->edges,                 // max_out_queue
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                false);
        }

        //if (/*DEBUG &&*/ (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
        //cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

        //frontier_attribute.queue_index++;
        //frontier_attribute.selector ^= 1;

        //util::MemsetAddKernel<<<128, 128>>>(problem->data_slices[0]->d_labels, 1, graph_slice->nodes);

        if (false) {//(INSTRUMENT || DEBUG) {
            //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
            //enactor_stats->total_queued += frontier_attribute->queue_length;
            if (Enactor::DEBUG) ShowDebugInfo<Problem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_filter", stream);
            if (Enactor::INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                    enactor_stats->filter_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false, stream)) return;
            }    
        }    
    }

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
              int             grid_size,
              int             block_size,
              size_t          shared_size,
              cudaStream_t    stream,
              SizeT           &num_elements,
        const VertexId* const keys_in,
        util::Array1D<SizeT, VertexId>*       keys_out,
        const size_t          array_size,
              char*           array,
              DataSlice*      data_slice)
    {
        bool over_sized = false;
        Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId>(
            "queue1", num_elements, keys_out, over_sized, -1, -1, -1);
        Expand_Incoming_Backward
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            keys_out->GetPointer(util::DEVICE),
            array_size,
            array);
    }

    static cudaError_t Compute_OutputLength(
        FrontierAttribute<SizeT> *frontier_attribute,
        SizeT       *d_offsets,
        VertexId    *d_indices,
        VertexId    *d_in_key_queue,
        util::Array1D<SizeT, SizeT>       *partitioned_scanned_edges,
        SizeT        max_in,
        SizeT        max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false)
    {
        cudaError_t retval = cudaSuccess;
        bool over_sized = false;
        if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
            "scanned_edges", frontier_attribute->queue_length, partitioned_scanned_edges, over_sized, -1, -1, -1, false)) return retval;
        retval = gunrock::oprtr::advance::ComputeOutputLength
            <AdvanceKernelPolicy, Problem, BackwardFunctor>(
            frontier_attribute,
            d_offsets,
            d_indices,
            d_in_key_queue,
            partitioned_scanned_edges->GetPointer(util::DEVICE),
            max_in,
            max_out,
            context,
            stream,
            ADVANCE_TYPE,
            express); 
        return retval;
    }

    static void Iteration_Change(long long &iterations)
    {
        iterations--;
    }

    static bool Stop_Condition(
        EnactorStats *enactor_stats,
        FrontierAttribute<SizeT> *frontier_attribute,
        util::Array1D<SizeT, DataSlice> *data_slice,
        int num_gpus)
    {
        //printf("Backward Stop checked\n");fflush(stdout);
        for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
        if (enactor_stats[gpu].retval!=cudaSuccess)
        {    
            printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
            return true;
        }    
        if (All_Done(enactor_stats, frontier_attribute, data_slice, num_gpus))
        {
            for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
            if (enactor_stats[gpu].iteration>1) {
                //printf("gpu %d iteration=%lld\n",gpu,enactor_stats[gpu].iteration);fflush(stdout);
                return false;
            }
            return true;
        } else return false;
    }

    static void Check_Queue_Size(
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        //util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        //DataSlice                     *data_slice,
        //DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice
        //util::CtaWorkProgressLifetime *work_progress,
        //ContextPtr                     context,
        //cudaStream_t                   stream
        )    
    {    
        return;
        /*bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                thread_num, iteration, peer_,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);fflush(stdout);

        if (enactor_stats->retval = 
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) return; 
        if (enactor_stats->retval = 
            Check_Size<true, SizeT, VertexId > ("queue3", graph_slice->nodes+2, &frontier_queue->keys  [selector  ], over_sized, thread_num, iteration, peer_, true )) return; 
        if (Problem::USE_DOUBLE_BUFFER)
        {    
            if (enactor_stats->retval = 
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector^1], over_sized, thread_num, iteration, peer_, false)) return; 
            if (enactor_stats->retval = 
                Check_Size<true, SizeT, Value> ("queue3", graph_slice->nodes+2, &frontier_queue->values[selector  ], over_sized, thread_num, iteration, peer_, true )) return; 
        } */   
    } 
};


    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BcEnactor>
    static CUT_THREADPROC BCThread(
        void * thread_data_)
    {
        typedef typename BcEnactor::Problem    Problem;
        typedef typename BcEnactor::SizeT      SizeT;
        typedef typename BcEnactor::VertexId   VertexId;
        typedef typename BcEnactor::Value      Value;
        typedef typename Problem::DataSlice  DataSlice;
        typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
        typedef ForwardFunctor<VertexId, SizeT, Value, Problem> BcFFunctor;
        typedef BackwardFunctor<VertexId, SizeT, Value, Problem> BcBFunctor;
        
        ThreadSlice  *thread_data         =  (ThreadSlice*) thread_data_;
        Problem      *problem             =  (Problem*)     thread_data->problem;
        BcEnactor    *enactor             =  (BcEnactor*)   thread_data->enactor;
        util::cpu_mt::CPUBarrier
                     *cpu_barrier         =   thread_data->cpu_barrier;
        int           num_gpus            =   problem     -> num_gpus;
        int           thread_num          =   thread_data -> thread_num;
        int           gpu_idx             =   problem     -> gpu_idx            [thread_num] ;
        DataSlice    *data_slice          =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
        util::Array1D<SizeT, DataSlice>
                     *s_data_slice        =   problem     -> data_slices;
        GraphSlice   *graph_slice         =   problem     -> graph_slices       [thread_num] ;
        FrontierAttribute<SizeT>
                     *frontier_attribute  = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
        EnactorStats *enactor_stats       = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
        EnactorStats *s_enactor_stats     = &(enactor     -> enactor_stats      [0                    ]);

        do {
            if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
            thread_data->stats = 1;
            while (thread_data->stats !=2) sleep(0);
            thread_data->stats = 3;

            for (int peer_=0;peer_<num_gpus;peer_++)
            {
                frontier_attribute[peer_].queue_index  = 0;        // Work queue index
                frontier_attribute[peer_].queue_length = peer_==0 ? thread_data -> init_size : 0; //? 
                frontier_attribute[peer_].selector     = frontier_attribute[peer_].queue_length == 0? 0:1;
                frontier_attribute[peer_].queue_reset  = true;
                enactor_stats     [peer_].iteration    = 0;
            }
           
            if (num_gpus>1)
            {
                data_slice->vertex_associate_orgs[0]=data_slice->labels.GetPointer(util::DEVICE);
                data_slice->vertex_associate_orgs[1]=data_slice->preds.GetPointer(util::DEVICE);
                data_slice->value__associate_orgs[0]=data_slice->sigmas.GetPointer(util::DEVICE);
                data_slice->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
                data_slice->value__associate_orgs.Move(util::HOST, util::DEVICE); 
            }
            gunrock::app::Iteration_Loop
                <2, 1, BcEnactor, BcFFunctor, Forward_Iteration<AdvanceKernelPolicy, FilterKernelPolicy, BcEnactor> > (thread_data);
            if (BcEnactor::DEBUG) util::cpu_mt::PrintMessage("Forward phase finished.", thread_num, enactor_stats->iteration);

            if (num_gpus>1)
            {
                data_slice->sigmas.Move(util::DEVICE, util::HOST);
                data_slice->labels.Move(util::DEVICE, util::HOST);
                //CPU_Barrier;
                util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[0],thread_num);
                long max_iteration=0;
                for (int gpu=0;gpu<num_gpus;gpu++)
                {
                    if (s_enactor_stats[gpu*num_gpus].iteration>max_iteration)
                        max_iteration=s_enactor_stats[gpu*num_gpus].iteration;
                }
                //printf("thread_num = %d, iteration = %d, max_iteration = %d\n", thread_num, enactor_stats[0].iteration, max_iteration);fflush(stdout);
                while (data_slice->forward_queue_offsets[0].size()<max_iteration)
                {
                    data_slice->forward_queue_offsets[0].push_back(data_slice->forward_queue_offsets[0].back());
                    //enactor_stats[0].iteration++;
                }
                enactor_stats[0].iteration=max_iteration; 
                for (VertexId node=0;node<graph_slice->in_counter[0];node++)
                {
                    for (SizeT i=graph_slice->backward_offset[node];i<graph_slice->backward_offset[node+1];i++)
                    {
                        int peer = graph_slice->backward_partition[i];
                        if (peer<=thread_num) peer--;
                        int _node = graph_slice->backward_convertion[i];
                        //printf("%d,%d -> %d,%d : %d,%f -> %d,%f\n", thread_num, node, peer, _node, s_data_slice[peer]->sigmas[_node], s_data_slice[peer]->labels[_node], data_slice->sigmas[node], data_slice->labels[node]);fflush(stdout);
                        s_data_slice[peer]->sigmas[_node]=data_slice->sigmas[node];
                        s_data_slice[peer]->labels[_node]=data_slice->labels[node];
                        //printf("(%f, %d) %d @ %d -> %d @ %d\n", data_slice->sigmas[node], data_slice->labels[node], _node, peer, node, thread_num);
                    }
                }
                //CPU_Barrier;
                util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[1],thread_num);
                //util::cpu_mt::PrintCPUArray<SizeT, Value>("sigmas", data_slice->sigmas.GetPointer(util::HOST), graph_slice->nodes, thread_num);
                //util::cpu_mt::PrintCPUArray<SizeT, VertexId>("labels", data_slice->labels.GetPointer(util::HOST), graph_slice->nodes, thread_num);
                for (int i=0;i<num_gpus;i++)
                for (int j=0;j<4;j++)
                for (int k=0;k<4;k++)
                    data_slice->events_set[j][i][k]=false;
                data_slice->sigmas.Move(util::HOST, util::DEVICE);
                data_slice->labels.Move(util::HOST, util::DEVICE);
                data_slice->value__associate_orgs[0]=data_slice->deltas.GetPointer(util::DEVICE);
                data_slice->value__associate_orgs[1]=data_slice->bc_values.GetPointer(util::DEVICE);
                data_slice->value__associate_orgs.Move(util::HOST, util::DEVICE);
            } else {
                //util::cpu_mt::PrintGPUArray("sigmas", data_slice->sigmas.GetPointer(util::DEVICE), graph_slice->nodes, thread_num);
                //util::cpu_mt::PrintGPUArray("labels", data_slice->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num);
            }

            
            //printf("thread_num = %d, size = %d: ", thread_num, data_slice->forward_queue_offsets[0].size());//fflush(stdout);
            //for (int i=0;i<data_slice->forward_queue_offsets[0].size();i++)
            //    printf(", %d",data_slice ->forward_queue_offsets[0].at(i));
            //printf("\n");fflush(stdout);

            gunrock::app::Iteration_Loop
                <0, 2, BcEnactor, BcBFunctor, Backward_Iteration<AdvanceKernelPolicy, FilterKernelPolicy, BcEnactor> > (thread_data);
            if (BcEnactor::DEBUG) util::cpu_mt::PrintMessage("Backward phase finished.", thread_num, enactor_stats->iteration);
       } while (0);

       thread_data->stats=4;
       CUT_THREADEND;
    }

/**
 * @brief BC problem enactor class.
 *
 * @tparam INSTRUMENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<typename _Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK>
class BCEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
     // Members
    _Problem    *problem      ;
    ThreadSlice *thread_slices;
    CUTThread   *thread_Ids   ;
    util::cpu_mt::CPUBarrier *cpu_barrier;

    // Methods
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

   /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BCEnactor constructor
     */
    BCEnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(VERTEX_FRONTIERS, num_gpus, gpu_idx)
    {
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
        cpu_barrier   = NULL;
    }

    /**
     *  @brief BCenactor destructor
     */
    virtual ~BCEnactor()
    {
        cutWaitForThreads(thread_Ids, this->num_gpus);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        problem = NULL;
        if (cpu_barrier!=NULL)
        {
            util::cpu_mt::DestoryBarrier(&cpu_barrier[0]);
            util::cpu_mt::DestoryBarrier(&cpu_barrier[1]);
            delete[] cpu_barrier;cpu_barrier=NULL;
        }
    }

    /**
     * @brief Obtain statistics about the last BC search enacted.
     *
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0; 
        total_queued = 0;
        //search_depth = 0;
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {   
            if (this->num_gpus!=1)
                if (util::SetDevice(this->gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            for (int peer=0; peer< this->num_gpus; peer++)
            {
                EnactorStats *enactor_stats_ = this->enactor_stats + gpu * this->num_gpus + peer;
                enactor_stats_ -> total_queued.Move(util::DEVICE, util::HOST);
                total_queued += enactor_stats_ -> total_queued[0];
                /*if (this->enactor_stats[gpu].iteration > search_depth) 
                    search_depth = this->enactor_stats[gpu].iteration;*/
                total_lifetimes += enactor_stats_ ->total_lifetimes;
                total_runtimes  += enactor_stats_ ->total_runtimes;
            }
        }   
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    template<
        typename AdvanceKernelPolity,
        typename FilterKernelPolicy>
    cudaError_t InitBC(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 512,
        bool        size_check    = true)
    {
        cudaError_t retval = cudaSuccess;
        cpu_barrier = new util::cpu_mt::CPUBarrier[2];
        cpu_barrier[0]=util::cpu_mt::CreateBarrier(this->num_gpus);
        cpu_barrier[1]=util::cpu_mt::CreateBarrier(this->num_gpus);
        // Lazy initialization
        if (retval = EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Init(problem,
                                       max_grid_size,
                                       AdvanceKernelPolity::CTA_OCCUPANCY,
                                       FilterKernelPolicy::CTA_OCCUPANCY)) return retval;
        
        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].cpu_barrier  = cpu_barrier;
            thread_slices[gpu].thread_num   = gpu;
            thread_slices[gpu].problem      = (void*)problem;
            thread_slices[gpu].enactor      = (void*)this;
            thread_slices[gpu].context      =&(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats        = -1;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(BCThread<
                    AdvanceKernelPolity, FilterKernelPolicy, 
                    BCEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }
        return retval;
    }

    cudaError_t Reset()
    {
        return EnactorBase<SizeT,DEBUG,SIZE_CHECK>::Reset();
    }

    /** @} */

    /**
     * @brief Enacts a brandes betweenness centrality computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for filter operator.
     * @tparam BCProblem BC Problem type.
     * @tparam ForwardFunctor Forward Functor type used in the forward sigma computing pass.
     * @tparam BackwardFunctor Backward Functor type used in the backward bc value accumulation pass.
     * @param[in] problem BCProblem object.
     * @param[in] src Source node for BC. -1 to compute BC value for each node.
     * @param[in] max_grid_size Max grid size for BC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactBC(VertexId    src)
    {
        cudaError_t              retval         = cudaSuccess;

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
                {
                    //printf("src = %d gpu = %d\n", src, gpu);fflush(stdout);
                    thread_slices[gpu].init_size=1;
                } else thread_slices[gpu].init_size=0;
                //this->frontier_attribute[gpu*this->num_gpus].queue_length = thread_slices[gpu].init_size;
            }

            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=1) sleep(0);
                thread_slices[gpu].stats=2;
            }
            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=4) sleep(0);
            }

            for (int gpu=0;gpu< this->num_gpus;gpu++)
                if (this->enactor_stats[gpu].retval!=cudaSuccess) 
                {retval=this->enactor_stats[gpu].retval;break;}
        } while (0);
        if (this->DEBUG) printf("\nGPU BC Done.\n");
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
     * @param[in] problem Pointer to BCProblem object.
     * @param[in] src Source node for BC. -1 indicates computing BC value for all nodes.
     * @param[in] max_grid_size Max grid size for BC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename BCProblem>
    cudaError_t Enact(VertexId    src)
    {
        int min_sm_version = -1; 
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
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
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                 // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                AdvanceKernelPolicy;

            return EnactBC<AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    cudaError_t Init(
        ContextPtr *context,
        Problem *problem,
        int max_grid_size = 512,
        bool size_check = true)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version ==-1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
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
                Problem,                           // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                 // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                AdvanceKernelPolicy;
           
            return InitBC<AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size, size_check);
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
