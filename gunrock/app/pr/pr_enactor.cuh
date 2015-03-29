// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

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

//#include <cub/cub.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace pr {

    template <typename Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK> class Enactor;

    template <typename DataSlice>
    __global__ void Print_Const (
        const DataSlice* const data_slice)
    {
        printf("delta = %f, threshold = %f, src_node = %d\n",
                data_slice->delta, data_slice->threshold, data_slice->src_node);
    }

    template <
        typename VertexId,
        typename SizeT>
    __global__ void Mark_Queue_R0D (
        const SizeT           num_elements,
        const VertexId* const keys_in,
        const SizeT*    const degrees,
              SizeT*          marker)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        VertexId x = blockIdx.x * blockDim.x + threadIdx.x;

        while ( x < num_elements)
        {
            VertexId key = keys_in[x];
            //if (degrees[key] == 0) printf("d[%d @ %d]==0 \t", key, x);
            marker[x] = degrees[key]==0? 1 :0;
            x += STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT>
    __global__ void Make_Queue_R0D (
        const SizeT           num_elements,
        const VertexId* const keys_in,
        const SizeT*    const marker,
              VertexId*       keys_out)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        VertexId x = blockIdx.x * blockDim.x + threadIdx.x;

        while (x < num_elements)
        {
            SizeT Mx = marker[x];
            if ((x!=0 && marker[x-1]!=Mx)
               ||(x==0 && Mx==1))
            {
                keys_out[Mx-1] = keys_in[x];
            }
            x += STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT,
        typename Value,
        int      NUM_VERTEX_ASSOCIATES,
        int      NUM_VALUE__ASSOCIATES>
    __global__ void Expand_Incoming_R0D (
        const SizeT           num_elements,
        const VertexId* const keys_in,
              SizeT*          degrees)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
        while (x < num_elements)
        {
            VertexId key = keys_in[x];
            degrees[key] = 0;
            x += STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT>
    __global__ void Clear_Zero_R0D (
        const SizeT        num_elements,
        const SizeT* const degrees_curr,
              SizeT*       degrees_next)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
        while (x < num_elements)
        {
            if (degrees_curr[x] == 0)
                degrees_next[x] = -1;
            x += STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT,
        typename Value,
        int      NUM_VERTEX_ASSOCIATES,
        int      NUM_VALUE__ASSOCIATES>
    __global__ void Expand_Incoming_PR (
        const SizeT           num_elements,
        const VertexId* const keys_in,
        const size_t          array_size,
              char*           array)
    {
        extern __shared__ char s_array[];
        const SizeT STRIDE = gridDim.x * blockDim.x;
        size_t offset = 0;
        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        Value** s_value__associate_in  = (Value**)&(s_array[offset]);
        offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        Value** s_value__associate_org = (Value**)&(s_array[offset]);
        SizeT x = threadIdx.x;
        while (x < array_size)
        {
            s_array[x] = array[x];
            x += blockDim.x;
        }
        __syncthreads();

        x = blockIdx.x * blockDim.x + threadIdx.x;
        while (x < num_elements)
        {
            VertexId key = keys_in[x];
            Value old_value=atomicAdd(s_value__associate_org[0] + key, s_value__associate_in[0][x]);
            //if (key == 0 || key == 1) printf("rank[%d] = %f + %f \t", key, old_value, s_value__associate_in[0][x]);
            x+=STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT>
    __global__ void Assign_Marker_PR(
        const SizeT     num_elements,
        const int       num_gpus,
        const SizeT*    markers,
        const int*      partition_table,
              SizeT**   key_markers)
    {
        extern __shared__ SizeT* s_marker[];
        int   gpu = 0;
        SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
        const SizeT STRIDE = gridDim.x * blockDim.x;
        if (threadIdx.x < num_gpus)
            s_marker[threadIdx.x] = key_markers[threadIdx.x];
        __syncthreads();

        while (x < num_elements)
        {
            //gpu = num_gpus;
            gpu = partition_table[x];
            if (markers[x] != 1 && gpu != 0)
            {
                gpu = num_gpus;
            } 
            for (int i=0; i<num_gpus; i++)
                s_marker[i][x] = (i==gpu)?1:0;
            x+=STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT>
    __global__ void Assign_Keys_PR (
        const SizeT          num_elements,
        const int            num_gpus,
        const int*           partition_table,
        const SizeT*         markers,
              SizeT**        keys_markers,
              VertexId**     keys_outs)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

        while (x < num_elements)
        {
            int gpu = partition_table[x];
            if (markers[x] == 1 || gpu == 0)
            {
                //if (gpu > 0)
                //{
                    SizeT pos = keys_markers[gpu][x]-1;
                    //printf("keys_outs[%d][%d] <- %d \t", gpu, pos, x);
                    keys_outs[gpu][pos] = x;
                //}
            }
            x+=STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT,
        typename Value>
    __global__ void Assign_Values_PR (
        const SizeT           num_elements,
        const VertexId* const keys_out,
        const Value*    const rank_next,
              Value*          rank_out)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        SizeT x = blockIdx.x * blockDim.x + threadIdx.x;

        while (x < num_elements)
        {
            VertexId key = keys_out[x];
            rank_out[x] = rank_next[key];
            x+=STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT,
        typename Value>
    __global__ void Expand_Incoming_Final (
        const SizeT num_elements,
        const VertexId* const keys_in,
        const Value*    const ranks_in,
              Value*          ranks_out)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        SizeT x = blockIdx.x * blockDim.x + threadIdx.x;
        while (x < num_elements)
        {
            VertexId key = keys_in[x];
            ranks_out[key] = ranks_in[x];
            x+=STRIDE;
        }
    }

template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct R0DIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false, //HAS_SUBQ
    true,  //HAS_FULLQ
    false, //BACKWARD
    true,  //FORWARD
    false> //UPDATE_PREDECESSORS
{
public:
    typedef typename Enactor::SizeT      SizeT     ;    
    typedef typename Enactor::Value      Value     ;    
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    typedef RemoveZeroDegreeNodeFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> RemoveZeroFunctor;

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
        //Print_Const<DataSlice><<<1,1,0,stream>>>(d_data_slice);
        if (enactor_stats->iteration == 0)
        {
            frontier_attribute->queue_reset  = true;
            frontier_attribute->selector     = 0;
            frontier_attribute->queue_index  = 0;
            frontier_attribute->queue_length = data_slice->num_gpus>1 ? data_slice->local_nodes : graph_slice->nodes;
        }
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
        //Print_Const<DataSlice><<<1,1,0,stream>>>(d_data_slice);
        SizeT num_valid_node = frontier_attribute->queue_length; 

        //util::DisplayDeviceResults(problem->graph_slices[0]->frontier_queues.d_keys[selector],
        //    num_elements);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys0", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray<SizeT, SizeT>("degrees0", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);

        bool over_sized = false;
        if (enactor_stats->retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT>(
            "scanned_edges", frontier_attribute->queue_length, scanned_edges, over_sized, thread_num, enactor_stats->iteration, peer_)) return;
        if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, stream)) return;
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, RemoveZeroFunctor>(
            //d_done,
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),// d_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
            (VertexId*)NULL,
            (VertexId*)NULL,
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes, //graph_slice->frontier_elements[frontier_attribute.selector],   // max_in_queue
            graph_slice->edges, //graph_slice->frontier_elements[frontier_attribute.selector^1], // max_out_queue
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);

        //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
        //      "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break; 

        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, RemoveZeroFunctor>
            <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats->iteration,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            //enactor_stats.num_gpus,
            frontier_attribute->queue_length,
            //d_done,
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
            NULL,
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
            d_data_slice,
            NULL,
            work_progress[0],
            frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
            enactor_stats->filter_kernel_stats);

        //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
        //      "filter::Kernel RemoveZeroFunctor failed", __FILE__, __LINE__)))
        //    break;

        Clear_Zero_R0D <SizeT, VertexId>
            <<<128, 128, 0, stream>>> (
            graph_slice->nodes,
            data_slice -> degrees.GetPointer(util::DEVICE),
            data_slice -> degrees_pong.GetPointer(util::DEVICE));

        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            data_slice->degrees.GetPointer(util::DEVICE),
            data_slice->degrees_pong.GetPointer(util::DEVICE), graph_slice->nodes);

        //util::DisplayDeviceResults(problem->data_slices[0]->d_degrees,
        //        graph_slice->nodes);

        frontier_attribute->queue_index++;
        frontier_attribute->selector^=1;
        if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, stream)) return;
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys1", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray<SizeT, SizeT>("degrees1", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);

        if (num_valid_node == frontier_attribute->queue_length || num_valid_node==0) data_slice->to_continue = false;
        else data_slice->to_continue = true;
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
            <AdvanceKernelPolicy, Problem, RemoveZeroFunctor>(
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

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
              int             grid_size,
              int             block_size,
              size_t          shared_size,
              cudaStream_t    stream,
              SizeT           &num_elements,
        const VertexId* const keys_in,
              VertexId*       keys_out,
        const size_t          array_size,
              char*           array,
              DataSlice*      data_slice)
    {
        Expand_Incoming_R0D
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            data_slice->degrees.GetPointer(util::DEVICE));
        num_elements = 0; 
    }

    static bool Stop_Condition(
        EnactorStats   *enactor_stats,
        FrontierAttribute<SizeT> *frontier_attribute,
        util::Array1D<SizeT, DataSlice> *data_slice,
        int num_gpus)
    {    
        //printf("CC Stop checked\n");fflush(stdout);
        for (int gpu = 0; gpu < num_gpus*num_gpus; gpu++)
        if (enactor_stats[gpu].retval != cudaSuccess)
        {    
            printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu%num_gpus, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
            return true;
        }    

        /*for (int gpu = 0; gpu< num_gpus*num_gpus; gpu++)
        if (enactor_stats[gpu].iteration == 0)
        {
            printf("enactor_stats[%d].iteration ==0\n", gpu);fflush(stdout);
            return false;
        }*/

        for (int gpu=0; gpu<num_gpus; gpu++)
            if (data_slice[gpu]->to_continue && frontier_attribute[gpu*num_gpus].queue_length !=0)
        {    
            //printf("data_slice[%d]->to_continue, frontier_attribute[%d].queue_length = %d\n", gpu, gpu*num_gpus, frontier_attribute[gpu*num_gpus].queue_length);fflush(stdout);
            return false;
        }    
     
        for (int gpu=0; gpu<num_gpus; gpu++)
        for (int peer=1; peer<num_gpus; peer++)
        for (int i=0; i<2; i++) 
        if (data_slice[gpu]->in_length[i][peer]!=0)
        {    
            //printf("data_slice[%d]->in_length[%d][%d] = %d\n", gpu, i, peer, data_slice[gpu]->in_length[i][peer]);fflush(stdout);
            return false;
        }    

        for (int gpu=0; gpu<num_gpus; gpu++)
        for (int peer=1; peer<num_gpus; peer++)
        if (data_slice[gpu]->out_length[peer]!=0) 
        {    
            //printf("data_slice[%d]->out_length[%d] = %d\n", gpu, peer, data_slice[gpu]->out_length[peer]); fflush(stdout);
            return false;
        }    
        //printf("CC to stop\n");fflush(stdout);
        return true;
    }    

    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (num_elements == 0)
        {
            for (int peer_ =0; peer_<num_gpus; peer_++)
                data_slice[0]->out_length[peer_] = 0;
            return;
        }
 
        int block_size = 256;
        int grid_size  = num_elements / block_size;
        int peer_      = 0;
        if ((num_elements % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size = 512;
       
        //util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice[0]->markers.GetPointer(util::DEVICE), 0, num_elements); 
        Mark_Queue_R0D <VertexId, SizeT>
            <<<grid_size, block_size, 0, stream>>> (
            num_elements,
            frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
            data_slice[0] -> degrees.GetPointer(util::DEVICE),
            data_slice[0] -> markers.GetPointer(util::DEVICE));
        //util::cpu_mt::PrintGPUArray("markers", data_slice[0]->markers.GetPointer(util::DEVICE), num_elements, thread_num, enactor_stats->iteration, -1, stream);

        Scan<mgpu::MgpuScanTypeInc>(
            (int*)data_slice[0] -> markers.GetPointer(util::DEVICE),
            num_elements,
            (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
            (int*)data_slice[0] -> markers.GetPointer(util::DEVICE),
            context[0]);

        Make_Queue_R0D <VertexId, SizeT>
            <<<grid_size, block_size, 0, stream>>> (
            num_elements,
            frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
            data_slice[0]->markers.GetPointer(util::DEVICE),
            data_slice[0]->keys_out[1].GetPointer(util::DEVICE));

        if (!Enactor::SIZE_CHECK)
            util::MemsetCopyVectorKernel <<<grid_size, block_size, 0, stream>>>(
                data_slice[0]->frontier_queues[0].keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                num_elements);

        cudaMemcpyAsync(&data_slice[0]->out_length[1], data_slice[0]->markers.GetPointer(util::DEVICE) + num_elements -1, sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
        //printf("num_lements = %d data_slice[%d]->out_length[1] = %d\n", num_elements, thread_num, data_slice[0]->out_length[1]);fflush(stdout);
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStramSynchronize failed", __FILE__, __LINE__)) return;
        //printf("num_lements = %d data_slice[%d]->out_length[1] = %d\n", num_elements, thread_num, data_slice[0]->out_length[1]);fflush(stdout);
        for (peer_ = 2; peer_ < num_gpus; peer_++)
            data_slice[0]->out_length[peer_] = data_slice[0]->out_length[1];
        data_slice[0]->out_length[0] = frontier_attribute->queue_length; 
    }

}; // end R0DIteration

template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct PRIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false, //HAS_SUBQ
    true,  //HAS_FULLQ
    false, //BACKWARD
    true,  //FORWARD
    false> //UPDATE_PREDECESSORS
{
public:
    typedef typename Enactor::SizeT      SizeT     ;    
    typedef typename Enactor::Value      Value     ;    
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice     <SizeT, VertexId, Value> GraphSlice;
    typedef PRFunctor      <VertexId, SizeT, Value, Problem> PrFunctor;
    typedef PRMarkerFunctor<VertexId, SizeT, Value, Problem> PrMarkerFunctor;

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
        //printf("Full queue start.\n");fflush(stdout); 
        //Print_Const<DataSlice><<<1,1,0,stream>>>(d_data_slice);
        if (enactor_stats -> iteration != 0)
        {
            frontier_attribute->queue_length = data_slice -> edge_map_queue_len;

            //printf("Filter start.\n");fflush(stdout); 
             // filter kernel
            gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, PrFunctor>
            <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                enactor_stats->iteration,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                NULL,
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);

            //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
            //cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates     

            //printf("Filter end.\n");fflush(stdout); 
            //enactor_stats->iteration++;
            frontier_attribute->queue_index++;

            if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, stream)) return;
            //num_elements = queue_length;

            //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,
            //    graph_slice->nodes);
            //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
            //    graph_slice->nodes);

            //swap rank_curr and rank_next
            util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
                data_slice->rank_curr.GetPointer(util::DEVICE),
                data_slice->rank_next.GetPointer(util::DEVICE), 
                graph_slice->nodes);
            util::MemsetKernel<<<128, 128, 0, stream>>>(
                data_slice->rank_next.GetPointer(util::DEVICE),
                (Value)0.0, graph_slice->nodes);

            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__));
            data_slice->PR_queue_length = frontier_attribute->queue_length;
            //printf("queue_length = %d\n", frontier_attribute->queue_length);fflush(stdout);
            if (false) {//if (INSTRUMENT || DEBUG) {
                //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
                enactor_stats->total_queued += frontier_attribute->queue_length;
                //if (DEBUG) printf(", %lld", (long long) frontier_attribute.queue_length);
                if (Enactor::INSTRUMENT) {
                    if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                        enactor_stats->filter_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes,
                        false, stream)) return;
                }
            }
        }

        //if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index, edge_map_queue_len)) return;
        frontier_attribute->queue_length = data_slice->edge_map_queue_len;
        //util::cpu_mt::PrintGPUArray<SizeT, SizeT>("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray<SizeT, Value>("ranks", data_slice->rank_curr.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray("keys", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);

        //printf("Advance start.\n");fflush(stdout); 
        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, PrFunctor>(
            //d_done,
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
            (VertexId*)NULL,
            (VertexId*)NULL,
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes,  //graph_slice->frontier_elements[frontier_attribute.selector],  // max_in_queue
            graph_slice->edges,  //graph_slice->frontier_elements[frontier_attribute.selector^1],// max_out_queue
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);
        //printf("Advance end.\n");fflush(stdout); 

        //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
        //cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates 

        /*if (Enactor::DEBUG) {
            if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, stream)) return;
        }

        if (Enactor::INSTRUMENT) {
            if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                enactor_stats->advance_grid_size,
                enactor_stats->total_runtimes,
                enactor_stats->total_lifetimes, false, stream)) return;
        }*/

        //if (done[0] == 0) break; 
        
        //if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index, edge_map_queue_len)) return;

        //if (done[0] == 0 || frontier_attribute.queue_length == 0 || enactor_stats.iteration > max_iteration) break;

        //if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);
    }

    static cudaError_t Compute_OutputLength(
        FrontierAttribute<SizeT> *frontier_attribute,
        SizeT       *d_offsets,
        VertexId    *d_indices,
        VertexId    *d_in_key_queue,
        util::Array1D<SizeT,SizeT>       *partitioned_scanned_edges,
        SizeT        max_in,
        SizeT        max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false)
    {   
        //printf("Compute_OutputLength start.\n");fflush(stdout);
        cudaError_t retval = cudaSuccess;
        if (AdvanceKernelPolicy::ADVANCE_MODE ==  gunrock::oprtr::advance::TWC_FORWARD) 
        {
            //return retval;
        } else {
            bool over_sized = false;
            if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
                "scanned_edges", frontier_attribute->queue_length, partitioned_scanned_edges, over_sized, -1, -1, -1, false)) return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, PrFunctor>(
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
        }
        //printf("Compute_OutputLength end.\n");fflush(stdout); 
        return retval;
    }    

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
              int             grid_size,
              int             block_size,
              size_t          shared_size,
              cudaStream_t    stream,
              SizeT           &num_elements,
        const VertexId* const keys_in,
              VertexId*       keys_out,
        const size_t          array_size,
              char*           array,
              DataSlice*      data_slice)
    {
        Expand_Incoming_PR
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            array_size,
            array);
        num_elements = 0; 
    }

    static bool Stop_Condition (
        EnactorStats                    *enactor_stats,
        FrontierAttribute<SizeT>        *frontier_attribute,
        util::Array1D<SizeT, DataSlice> *data_slice,
        int num_gpus)
    {
        bool all_zero = true;
        for (int gpu = 0; gpu < num_gpus*num_gpus; gpu++)
        if (enactor_stats[gpu].retval != cudaSuccess)
        {    
            //printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu%num_gpus, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
            return true;
        }  

        for (int gpu =0; gpu < num_gpus; gpu++)
        if (data_slice[gpu]->PR_queue_length > 0) 
        {
            //printf("data_slice[%d].PR_queue_length = %d\n", gpu, data_slice[gpu]->PR_queue_length);
            all_zero = false;
        }
        if (all_zero) return true;

        for (int gpu =0; gpu < num_gpus; gpu++)
        if (enactor_stats[gpu * num_gpus].iteration < data_slice[0]->max_iter)
        {
            //printf("enactor_stats[%d].iteration = %lld\n", gpu, enactor_stats[gpu * num_gpus].iteration);
            return false;    
        } 

        return true;
    }    

    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        //printf("Make_Output entered\n");fflush(stdout);
        int peer_      = 0;
        int block_size = 512;
        int grid_size  = graph_slice->nodes / block_size;
        if ((graph_slice->nodes % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size = 512;

        if (num_gpus > 1 && enactor_stats->iteration==0)
        {
            util::MemsetKernel<<<grid_size, block_size, 0, stream>>>(data_slice[0]->markers.GetPointer(util::DEVICE), (SizeT)0, graph_slice->nodes);
            frontier_attribute->queue_length = data_slice[0]->edge_map_queue_len;
            //util::cpu_mt::PrintGPUArray("keys", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, -1, stream);
            //util::cpu_mt::PrintGPUArray("row_offsets", graph_slice->row_offsets.GetPointer(util::DEVICE), graph_slice->nodes+1, thread_num, enactor_stats->iteration, -1, stream);
            //printf("Advance start.\n");fflush(stdout); 
            // Edge Map
            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, PrMarkerFunctor>(
                //d_done,
                enactor_stats[0],
                frontier_attribute[0],
                data_slice->GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                (VertexId*)NULL,
                (VertexId*)NULL,
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,  //graph_slice->frontier_elements[frontier_attribute.selector],  // max_in_queue
                graph_slice->edges,  //graph_slice->frontier_elements[frontier_attribute.selector^1],// max_out_queue
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V,
                false,
                true);
            //printf("Advance end.\n");fflush(stdout);
            //util::cpu_mt::PrintGPUArray("markers", data_slice[0]->markers.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, -1, stream);
            
            for (peer_ = 0; peer_<num_gpus; peer_++)
                util::MemsetKernel<<<128, 128, 0, stream>>> ( data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE), 0, graph_slice->nodes);
            Assign_Marker_PR<VertexId, SizeT>
                <<<grid_size, block_size, num_gpus * sizeof(SizeT*), stream>>> (
                graph_slice->nodes,
                num_gpus,
                data_slice[0]->markers.GetPointer(util::DEVICE),
                graph_slice->partition_table.GetPointer(util::DEVICE),
                data_slice[0]->keys_markers.GetPointer(util::DEVICE));
            //for (peer_ = 0; peer_<num_gpus;peer_++)
            //    util::cpu_mt::PrintGPUArray("keys_marker", data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, -1, stream);

            for (peer_ = 0; peer_<num_gpus;peer_++)
                Scan<mgpu::MgpuScanTypeInc>(
                    (int*)(data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE)),
                    graph_slice->nodes,
                    (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
                    (int*)(data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE)),
                    context[0]);
            //for (peer_ = 0; peer_<num_gpus;peer_++)
            //    util::cpu_mt::PrintGPUArray("keys_marker", data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, -1, stream);

            SizeT temp_length = data_slice[0]->out_length[0];
            if (graph_slice->nodes > 0) for (peer_ = 0; peer_<num_gpus; peer_++)
            {
                cudaMemcpyAsync(
                    &data_slice[0]->out_length[peer_], 
                    data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE) + (graph_slice->nodes -1),
                    sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
            } else {
                for (peer_ = 1; peer_<num_gpus; peer_++)
                    data_slice[0]->out_length[peer_] = 0;
            }
            if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;

            for (peer_ = 0; peer_<num_gpus; peer_++)
            {
                bool over_sized = false;
                if (peer_>1) data_slice[0]->keys_out[peer_] = data_slice[0]->temp_keys_out[peer_];
                if (enactor_stats->retval = Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId> (
                    "keys_out", data_slice[0]->out_length[peer_], &data_slice[0]->keys_out[peer_], over_sized, thread_num, enactor_stats->iteration, peer_)) return;
                if (peer_>0)
                    if (enactor_stats->retval = Check_Size<Enactor::SIZE_CHECK, SizeT, Value> (
                        "values_out", data_slice[0]->out_length[peer_], &data_slice[0]->value__associate_out[peer_][0], over_sized, thread_num, enactor_stats->iteration, peer_)) return;
                if (!over_sized) continue;
                data_slice[0]->keys_outs[peer_] = data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE);
                data_slice[0]->value__associate_outs[peer_][0] = data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE);
                data_slice[0]->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, -1, 0, stream);
            } 
            data_slice[0]->keys_outs.Move(util::HOST, util::DEVICE, -1, 0, stream);
            data_slice[0]->out_length[0] = temp_length;

            Assign_Keys_PR <VertexId, SizeT>
                <<<grid_size, block_size, num_gpus * sizeof(SizeT*) *2, stream>>> (
                graph_slice->nodes,
                num_gpus,
                graph_slice->partition_table.GetPointer(util::DEVICE),
                data_slice[0]->markers      .GetPointer(util::DEVICE),
                data_slice[0]->keys_markers .GetPointer(util::DEVICE),
                data_slice[0]->keys_outs    .GetPointer(util::DEVICE));
                
            //util::cpu_mt::PrintCPUArray("out_length", &data_slice[0]->out_length[0], num_gpus, thread_num, enactor_stats->iteration);
            //util::cpu_mt::PrintGPUArray("keys_out[1]", data_slice[0]->keys_out[1].GetPointer(util::DEVICE), data_slice[0]->out_length[1], num_gpus, thread_num, enactor_stats->iteration);
        }

        for (peer_ = 1; peer_ < num_gpus; peer_ ++)
        {
            Assign_Values_PR <VertexId, SizeT, Value>
                <<<grid_size, block_size, 0, stream>>> (
                data_slice[0]->out_length[peer_],
                data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE),
                data_slice[0]->rank_next.GetPointer(util::DEVICE),
                data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE));
        }
        frontier_attribute->selector = data_slice[0]->PR_queue_selector;
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
    }
 
}; 

    /**
     * @brief Enacts a page rank computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     * @tparam PRProblem PR Problem type.
     *
     * @param[in] problem PRProblem object.
     * @param[in] src Source node for PR.
     * @param[in] max_grid_size Max grid size for PR kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename PrEnactor>
    static CUT_THREADPROC PRThread(
        void * thread_data_)
    {
        typedef typename PrEnactor::Problem    Problem;
        typedef typename PrEnactor::SizeT      SizeT;
        typedef typename PrEnactor::VertexId   VertexId;
        typedef typename PrEnactor::Value      Value;
        typedef typename Problem::DataSlice    DataSlice;
        typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
        typedef PRFunctor<VertexId, SizeT, Value, Problem> PrFunctor;
        ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
        Problem      *problem            =  (Problem*)     thread_data->problem;
        PrEnactor    *enactor            =  (PrEnactor*)   thread_data->enactor;
        int           num_gpus           =   problem     -> num_gpus;
        int           thread_num         =   thread_data -> thread_num;
        int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
        DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
        GraphSlice   *graph_slice        =   problem     -> graph_slices       [thread_num] ;
        FrontierAttribute<SizeT>
                     *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
        EnactorStats *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

        do {
            printf("CCThread entered\n");fflush(stdout);
            if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
            thread_data->stats = 1;
            while (thread_data->stats !=2) sleep(0);
            thread_data->stats = 3;

            for (int peer_=0; peer_<num_gpus; peer_++)
            {
                frontier_attribute[peer_].queue_length  = peer_==0?data_slice->local_nodes : 0;
                frontier_attribute[peer_].queue_index   = 0;        // Work queue index
                frontier_attribute[peer_].selector      = 0;
                frontier_attribute[peer_].queue_reset   = true;
                enactor_stats     [peer_].iteration     = 0;
            }
            gunrock::app::Iteration_Loop
                <0, 0, PrEnactor, PrFunctor, R0DIteration<AdvanceKernelPolicy, FilterKernelPolicy, PrEnactor> > (thread_data);
            
            data_slice->PR_queue_selector = frontier_attribute[0].selector;
            for (int peer_=0; peer_<num_gpus; peer_++)
            {
                frontier_attribute[peer_].queue_reset = true;
                enactor_stats     [peer_].iteration   = 0;
            }
            if (num_gpus > 1)
            {
                data_slice->value__associate_orgs[0] = data_slice->rank_next.GetPointer(util::DEVICE);
                data_slice->value__associate_orgs.Move(util::HOST, util::DEVICE);
            }
            data_slice -> edge_map_queue_len = frontier_attribute[0].queue_length;

            // Step through PR iterations
            gunrock::app::Iteration_Loop
                <0, 1, PrEnactor, PrFunctor, PRIteration<AdvanceKernelPolicy, FilterKernelPolicy, PrEnactor> > (thread_data);
            
            if (thread_num > 0)
            {
                bool over_sized = false;
                if (enactor_stats->retval = Check_Size<PrEnactor::SIZE_CHECK, SizeT, Value>(
                    "values_out", data_slice->local_nodes, &data_slice->value__associate_out[1][0], over_sized, thread_num, enactor_stats->iteration, -1)) break;
                if (enactor_stats->retval = Check_Size<PrEnactor::SIZE_CHECK, SizeT, VertexId>(
                    "keys_out", data_slice->local_nodes, &data_slice->keys_out[1], over_sized, thread_num, enactor_stats->iteration, -1)) break;
                Assign_Values_PR <VertexId, SizeT, Value>
                    <<<128, 128, 0, data_slice->streams[0]>>> (
                    data_slice->local_nodes,
                    data_slice->keys_out[0].GetPointer(util::DEVICE),
                    data_slice->rank_curr.GetPointer(util::DEVICE),
                    data_slice->value__associate_out[1][0].GetPointer(util::DEVICE));
                util::MemsetCopyVectorKernel<<<128, 128, 0, data_slice->streams[0]>>> (
                    data_slice->keys_out[1].GetPointer(util::DEVICE),
                    data_slice->keys_out[0].GetPointer(util::DEVICE),
                    data_slice->local_nodes);
                enactor_stats->iteration++;
                PushNeibor <PrEnactor::SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice, 0, 1> (
                    thread_num,
                    0,
                    data_slice->local_nodes,
                    enactor_stats,
                    problem->data_slices [thread_num].GetPointer(util::HOST),
                    problem->data_slices [0         ].GetPointer(util::HOST),
                    problem->graph_slices[thread_num],
                    problem->graph_slices[0],
                    data_slice->streams[0]);
                Set_Record(data_slice, enactor_stats->iteration, 1, 0, data_slice->streams[0]);
                data_slice->final_event_set = true;
                //util::cpu_mt::PrintGPUArray("keys_out", data_slice->keys_out[1].GetPointer(util::DEVICE), data_slice->local_nodes, thread_num, enactor_stats->iteration, -1, data_slice->streams[0]); 
                //util::cpu_mt::PrintGPUArray("values_out", data_slice->value__associate_out[1][0].GetPointer(util::DEVICE), data_slice->local_nodes, thread_num, enactor_stats->iteration, -1, data_slice->streams[0]); 
            } else {
                int counter = 0;
                int *markers = new int [num_gpus];
                for (int peer=0; peer<num_gpus; peer++) markers[peer] = 0;
                while (counter < num_gpus-1)
                {
                    for (int peer=1; peer<num_gpus; peer++)
                    if (markers[peer] == 0 && problem->data_slices[peer]->final_event_set)
                    {
                        markers[peer] =1 ;
                        counter ++;
                        problem->data_slices[peer]->final_event_set = false;
                        int peer_iteration = enactor->enactor_stats[peer * num_gpus].iteration;
                        cudaStreamWaitEvent(data_slice->streams[peer], 
                            problem->data_slices[peer]->events[peer_iteration%4][1][0], 0);
                        Expand_Incoming_Final<VertexId, SizeT, Value>
                            <<<128, 128, 0, data_slice->streams[peer]>>> (
                            problem->data_slices[peer]->local_nodes,
                            data_slice->keys_in[peer_iteration%2][peer].GetPointer(util::DEVICE),
                            data_slice->value__associate_in[peer_iteration%2][peer][0].GetPointer(util::DEVICE),
                            data_slice->rank_curr.GetPointer(util::DEVICE));
                    }
                }
                for (int peer=1; peer<num_gpus; peer++)
                {
                    int peer_iteration = enactor->enactor_stats[peer * num_gpus].iteration;
                    if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(data_slice->streams[peer]),
                        "cudaStreamSynchronize failed", __FILE__, __LINE__)) break;
                    //util::cpu_mt::PrintGPUArray("keys_in", data_slice->keys_in[peer_iteration%2][peer].GetPointer(util::DEVICE), problem->data_slices[peer]->local_nodes, thread_num, peer_iteration, peer);
                    //util::cpu_mt::PrintGPUArray("ranks_in", data_slice->value__associate_in[peer_iteration%2][peer][0].GetPointer(util::DEVICE), problem->data_slices[peer]->local_nodes, thread_num, peer_iteration, peer);
                }
                // sort according to the rank of nodes
                util::CUBRadixSort<Value, VertexId>(
                    false, graph_slice->nodes,
                    data_slice->rank_curr.GetPointer(util::DEVICE),
                    data_slice->node_ids.GetPointer(util::DEVICE));
            }

            //if (d_scanned_edges) cudaFree(d_scanned_edges);
        
        } while(0); 

        printf("PR_Thread finished\n");fflush(stdout);
        thread_data->stats = 4;
        CUT_THREADEND;
    }


/**
 * @brief PR problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <
    typename _Problem,
    bool _INSTRUMENT,
    bool _DEBUG,
    bool _SIZE_CHECK>
class PREnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
    //Members
    _Problem    *problem      ;
    ThreadSlice *thread_slices;
    CUTThread   *thread_Ids   ;
    
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
     * @brief PREnactor constructor
     */
    PREnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(VERTEX_FRONTIERS, num_gpus, gpu_idx)
    {    
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
    }    

    /**  
     *  @brief PREnactor destructor
     */
    virtual ~PREnactor()
    {    
        cutWaitForThreads(thread_Ids, this->num_gpus);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        problem = NULL;
    }    

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last PR search enacted.
     *
     * @param[out] total_queued Total queued elements in PR kernel running.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        unsigned long long total_lifetimes = 0;
        unsigned long long total_runtimes  = 0;
        total_queued = 0;

        for (int gpu=0; gpu<this->num_gpus; gpu++)
        {
            if (this->num_gpus!=1)
                if (util::SetDevice(this->gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            total_queued    += this->enactor_stats[gpu].total_queued;
            total_lifetimes += this->enactor_stats[gpu].total_lifetimes;
            total_runtimes  += this->enactor_stats[gpu].total_runtimes;
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * \addtogroup PublicInterface
     * @{
     */
    template<
        typename AdvanceKernelPolity,
        typename FilterKernelPolicy>
    cudaError_t InitPR(
        ContextPtr  *context,
        Problem     *problem,
        //int         max_iteration,
        int         max_grid_size = 512)
    {
        cudaError_t retval = cudaSuccess;
        // Lazy initialization
        if (retval = EnactorBase <SizeT, DEBUG, SIZE_CHECK> ::Init(
            problem,
            max_grid_size,
            AdvanceKernelPolity::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        if (DEBUG) {
            printf("PR vertex map occupancy %d, level-grid size %d\n",
                        FilterKernelPolicy::CTA_OCCUPANCY, this->enactor_stats[0].filter_grid_size);
        }

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].thread_num   = gpu;
            thread_slices[gpu].problem      = (void*)problem;
            thread_slices[gpu].enactor      = (void*)this;
            thread_slices[gpu].context      =&(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats        = -1;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(PRThread<
                    AdvanceKernelPolity, FilterKernelPolicy,
                    PREnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }
        return retval;
    }

    cudaError_t Reset()
    {
        return EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Reset();
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactPR()
    {
        cudaError_t              retval         = cudaSuccess;

        do {
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
        if (this->DEBUG) printf("\nGPU PR Done.\n");
        return retval;
    }

    /**
     * @brief PR Enact kernel entry.
     *
     * @tparam PRProblem PR Problem type. @see PRProblem
     *
     * @param[in] problem Pointer to PRProblem object.
     * @param[in] src Source node for PR.
     * @param[in] max_grid_size Max grid size for PR kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        int   traversal_mode)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                1,                                  // MIN_CTA_OCCUPANCY
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
                Problem,                            // Problem data type
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

            if (traversal_mode == 1)
            {
                return EnactPR<
                    FWDAdvanceKernelPolicy, FilterKernelPolicy>();
            }
            else
            {
                return EnactPR<
                     LBAdvanceKernelPolicy, FilterKernelPolicy>();
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    cudaError_t Init(
            ContextPtr *context,
            Problem    *problem,
            int         traversal_mode,
            //int         max_iteration,
            int         max_grid_size = 512)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                1,                                  // MIN_CTA_OCCUPANCY
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
                Problem,                            // Problem data type
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

            if (traversal_mode == 1)
                return InitPR<FWDAdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, /*max_iteration,*/ max_grid_size);
            else return InitPR<LBAdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, /*max_iteration,*/ max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

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
