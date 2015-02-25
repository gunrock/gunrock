// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cc_enactor.cuh
 *
 * @brief CC Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>


namespace gunrock {
namespace app {
namespace cc {

    template <typename Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK> class Enactor;

    template <
        typename VertexId,
        typename SizeT,
        typename Value,
        int      NUM_VERTEX_ASSOCIATES,
        int      NUM_VALUE__ASSOCIATES>
    __global__ void Expand_Incoming_BothWay (
        const SizeT            num_elements,
        const VertexId* const  keys_in,
              VertexId*        keys_out,
        const size_t           array_size,
              char*            array)
    {
        extern __shared__ char s_array[];
        const SizeT STRIDE = gridDim.x * blockDim.x;
        size_t      offset                 = 0;
        VertexId**  s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
        //Value**     s_value__associate_in  = (Value**   )&(s_array[offset]);
        offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
        VertexId**  s_vertex_associate_org = (VertexId**)&(s_array[offset]);
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
            VertexId key = keys_in[x];
            //atomicMin(s_vertex_associate_org[0]+key, s_vertex_associate_in[0][x]);
            if (s_vertex_associate_in[0][x] < s_vertex_associate_org[0][key])
            {
                //if (to_track(key))
                //    printf("Expand_Incoming [%d]: %d->%d\n", key, s_vertex_associate_org[0][key], s_vertex_associate_in[0][x]);
                s_vertex_associate_org[0][key] = s_vertex_associate_in[0][x]; 
            }
            keys_out[x]=-1;
            x+=STRIDE;
        }
    }

    template <
        typename VertexId, 
        typename SizeT>
    __global__ void Mark_Difference_Queue (
        const SizeT           num_elements,
        //const VertexId* const keys_in,
        //      VertexId*       keys_out,
        const VertexId* const old_CID,
        const VertexId* const new_CID,
              SizeT*          marker)
    {
        VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
        const SizeT STRIDE = gridDim.x * blockDim.x;

        while (x<num_elements)
        {
            //VertexId key = keys_in[x]
            //if (to_track(x))
            //    printf("Mark_Diff marker[%d]: %d->%d, CID: %d->%d\n", x, marker[x], (old_CID[x]!=new_CID[x]? 1:0), old_CID[x], new_CID[x]);
            marker[x] = old_CID[x]!=new_CID[x]? 1:0;
            x+=STRIDE;
        }
    }

    template <
        typename VertexId,
        typename SizeT>
    __global__ void Make_Difference_Queue (
        const SizeT         num_elements,
        const SizeT* const  marker,
              VertexId*     keys_out)
    {
        VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
        const SizeT STRIDE = gridDim.x * blockDim.x;

        while (x<num_elements)
        {
            if ((x!=0 && marker[x-1]!=marker[x])
               ||(x==0 && marker[x]==1)) 
            {
                keys_out[marker[x]-1]=x;
                //if (to_track(x))
                //    printf("Make_Diff keys_out[%d] ->%d\n", marker[x]-1, x);
            }
            x+=STRIDE;
        }
    }

template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct CCIteration : public IterationBase <AdvanceKernelPolicy, FilterKernelPolicy, Enactor>
{
public:
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;

    static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = false;
    static const bool HAS_FULLQ  = true;
    static const bool BACKWARD   = true;
    static const bool FORWARD    = true;
    static const bool UPDATE_PREDECESSORS = false;

    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        GraphSlice                    *graph_slice,
        cudaStream_t                   stream)
    {
        typedef HookInitFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> HookInitFunctor;

        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(frontier_queue->keys[0].GetPointer(util::DEVICE), graph_slice->edges);
        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(frontier_queue->values[0].GetPointer(util::DEVICE), graph_slice->nodes);
        util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->marks.GetPointer(util::DEVICE), false, graph_slice->edges);
        //util::MemsetKernel<<<128, 128, 0, stream>>>(data_slice->masks.GetPointer(util::DEVICE), 0, graph_slice->nodes);

        //printf("queue set\n");fflush(stdout);
        if (data_slice->turn==0)
        {
            frontier_attribute->queue_index  = 0;
            frontier_attribute->selector     = 0;
            frontier_attribute->queue_length = graph_slice->edges;
            frontier_attribute->queue_reset  = true;
            //printf("HookInit begin, %d, %d, %p\n", frontier_queue->keys[frontier_attribute->selector].GetSize(), frontier_queue->keys[frontier_attribute->selector^1].GetSize(), data_slice->d_pointer);fflush(stdout);
            gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, HookInitFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                0,  //iteration, not used in CC
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),  // d_in_queue
                NULL,   //pred_queue, not used in CC
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),  // d_out_queue
                data_slice->d_pointer,
                NULL,   //d_visited_mask, not used in CC
                data_slice->work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);
            if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel Initial HookInit Operation failed", __FILE__, __LINE__))) return;
            //printf("HookInited\n");fflush(stdout);
        }

        if (data_slice->num_gpus > 1)
        { 
            if (data_slice->turn==0)
                util::MemsetIdxKernel<<<128, 128, 0, stream>>> (
                data_slice->old_c_ids.GetPointer(util::DEVICE),
                graph_slice->nodes);
            else util::MemsetCopyVectorKernel
                <<<128, 128, 0, stream>>>(
                data_slice->old_c_ids.GetPointer(util::DEVICE), 
                data_slice->component_ids.GetPointer(util::DEVICE), 
                graph_slice->nodes);
        }
        data_slice->turn++;
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
        typedef UpdateMaskFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> UpdateMaskFunctor;

        typedef HookMinFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> HookMinFunctor;

        typedef HookMaxFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> HookMaxFunctor;

        typedef PtrJumpFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> PtrJumpFunctor;

        typedef PtrJumpMaskFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> PtrJumpMaskFunctor;

        typedef PtrJumpUnmaskFunctor<
            VertexId,
            SizeT,
            VertexId,
            Problem> PtrJumpUnmaskFunctor;

        //bool has_change                  = false;
        // Pointer Jumping
        enactor_stats->iteration=0; 
        frontier_attribute->queue_index  = 0;
        frontier_attribute->selector     = 0;
        frontier_attribute->queue_length = graph_slice->nodes;
        frontier_attribute->queue_reset  = true;

        //util::cpu_mt::PrintGPUArray("0_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);
        // First Pointer Jumping Round
        data_slice->vertex_flag[0] = 0;
        while (!data_slice->vertex_flag[0]) {
            data_slice->vertex_flag[0] = 1;
            data_slice->vertex_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);

            gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, PtrJumpFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                0,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                NULL,
                frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->values[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                frontier_queue->values[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);
            if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel First Pointer Jumping Round failed", __FILE__, __LINE__))) return;

            frontier_attribute->queue_reset = false;
            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;
            enactor_stats->iteration++;
            data_slice->vertex_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);

            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
            //util::cpu_mt::PrintMessage("PtrJump finished", thread_num, data_slice->turn, enactor_stats->iteration);
            // Check if done
            if (data_slice->vertex_flag[0]) break;
            //else has_change = true;
        }
        //util::cpu_mt::PrintGPUArray("1_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);

        frontier_attribute->queue_index   = 0;        // Work queue index
        frontier_attribute->selector      = 0;
        frontier_attribute->queue_length  = graph_slice->nodes;
        frontier_attribute->queue_reset   = true;

        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, UpdateMaskFunctor>
            <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            0,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
            NULL,
            frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
            d_data_slice,
            NULL,
            work_progress[0],
            frontier_queue->values[frontier_attribute->selector  ].GetSize(),           // max_in_queue
            frontier_queue->values[frontier_attribute->selector^1].GetSize(),         // max_out_queue
            enactor_stats->filter_kernel_stats);
        if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel Update Mask Operation failed", __FILE__, __LINE__))) return;
        //util::cpu_mt::PrintGPUArray("2_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);
        //util::cpu_mt::PrintMessage("Update Mask finished", thread_num, data_slice->turn, enactor_stats->iteration);

        enactor_stats->iteration = 1;
        data_slice->edge_flag[0] = 0;
        while (!data_slice->edge_flag[0]) 
        {
            frontier_attribute->queue_index  = 0;        // Work queue index
            frontier_attribute->queue_length = graph_slice->edges;
            frontier_attribute->selector     = 0;
            frontier_attribute->queue_reset  = true;
            data_slice->edge_flag[0] = 1;
            data_slice->edge_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);

            /*if (enactor_stats->iteration & 3) {
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, HookMinFunctor>
                    <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                    0,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),  // d_in_queue
                    NULL,
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),  // d_out_queue
                    d_data_slice,
                    NULL,
                    work_progress[0],
                    frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                    enactor_stats->filter_kernel_stats);
            } else {*/
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, HookMaxFunctor>
                    <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                    0,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                    NULL,
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    d_data_slice,
                    NULL,
                    work_progress[0],
                    frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                    enactor_stats->filter_kernel_stats);
            //}
            if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel Hook Min/Max Operation failed", __FILE__, __LINE__))) return;
            //util::cpu_mt::PrintGPUArray("3_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);
            //util::cpu_mt::PrintMessage("HookMax finished", thread_num, data_slice->turn, enactor_stats->iteration);

            frontier_attribute->queue_reset = false;
            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;
            enactor_stats->iteration++;

            data_slice->edge_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);
            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
            // Check if done
            if (data_slice->edge_flag[0]) break; //|| enactor_stats->iteration>5) break;
            //else has_change = true;

            ///////////////////////////////////////////
            // Pointer Jumping 
            frontier_attribute->queue_index  = 0;
            frontier_attribute->selector     = 0;
            frontier_attribute->queue_length = graph_slice->nodes;
            frontier_attribute->queue_reset  = true;

            // First Pointer Jumping Round
            data_slice->vertex_flag[0] = 0;
            while (!data_slice->vertex_flag[0]) 
            {
                data_slice->vertex_flag[0] = 1;
                data_slice->vertex_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, PtrJumpMaskFunctor>
                    <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                    0,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    frontier_attribute->queue_length,
                    frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                    NULL,
                    frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                    d_data_slice,
                    NULL,
                    work_progress[0],
                    frontier_queue->values[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                    frontier_queue->values[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                    enactor_stats->filter_kernel_stats);
                if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel Pointer Jumping Mask failed", __FILE__, __LINE__))) return;

                frontier_attribute->queue_reset = false;
                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;

                data_slice->vertex_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);
                if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
                //util::cpu_mt::PrintMessage("Pointer Jumping Mask finished", thread_num, data_slice->turn, enactor_stats->iteration);
                // Check if done
                if (data_slice->vertex_flag[0]) break;
                //else has_change = true;
            }
            //util::cpu_mt::PrintGPUArray("4_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);

            frontier_attribute->queue_index  = 0;        // Work queue index
            frontier_attribute->selector     = 0;
            frontier_attribute->queue_length = graph_slice->nodes;
            frontier_attribute->queue_reset  = true;

            gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, PtrJumpUnmaskFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                0,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                NULL,
                frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->values[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                frontier_queue->values[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);
            if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel Pointer Jumping Unmask Operation failed", __FILE__, __LINE__))) return;
            //util::cpu_mt::PrintGPUArray("5_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);
            //util::cpu_mt::PrintMessage("Pointer Jumping Unmask finished", thread_num, data_slice->turn, enactor_stats->iteration);

            gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, UpdateMaskFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
                0,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                NULL,
                frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->values[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                frontier_queue->values[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);
            if (DEBUG && (enactor_stats->retval = util::GRError("filter::Kernel Update Mask Operation failed", __FILE__, __LINE__))) return;
            //util::cpu_mt::PrintMessage("Update Mask finished", thread_num, data_slice->turn, enactor_stats->iteration);

            ///////////////////////////////////////////
        }
        //util::cpu_mt::PrintGPUArray("6_cid", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, data_slice->turn, enactor_stats->iteration, stream);
        //util::cpu_mt::PrintMessage("Loop finished", thread_num, data_slice->turn, enactor_stats->iteration);

        if (data_slice->num_gpus > 1)
        {
            int grid_size = graph_slice -> nodes / 256 +1;
            if (grid_size > 512) grid_size = 512;
            Mark_Difference_Queue<VertexId, SizeT>
                <<<grid_size, 256, 0, stream>>> (
                graph_slice -> nodes,
                data_slice  -> old_c_ids.GetPointer(util::DEVICE),
                data_slice  -> component_ids.GetPointer(util::DEVICE),
                data_slice  -> CID_markers.GetPointer(util::DEVICE));

            Scan<mgpu::MgpuScanTypeInc>(
                (int*)data_slice->CID_markers.GetPointer(util::DEVICE),
                graph_slice ->nodes,
                (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
                (int*)data_slice->CID_markers.GetPointer(util::DEVICE),
                context[0]);

            Make_Difference_Queue<VertexId, SizeT>
                <<<grid_size, 256, 0, stream>>> (
                graph_slice->nodes,
                data_slice->CID_markers.GetPointer(util::DEVICE),
                frontier_queue->keys[0].GetPointer(util::DEVICE));

            frontier_attribute->selector = 0;
            frontier_attribute->queue_reset = 0;
            cudaMemcpyAsync(&frontier_attribute->queue_length, data_slice->CID_markers.GetPointer(util::DEVICE) + graph_slice->nodes -1, sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            work_progress->SetQueueLength(
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                false,
                stream);

            if (frontier_attribute->queue_length > 0)
            {
                data_slice -> has_change = true;
            } else data_slice -> has_change = false;
            enactor_stats->iteration = data_slice->turn;
            //printf("%d\t %d\t \t has_change = %s\n", thread_num, data_slice->turn, data_slice -> has_change? "true" : "false"); fflush(stdout);
            //util::cpu_mt::PrintGPUArray("change keys", frontier_queue->keys[0].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration);
            //util::cpu_mt::PrintGPUArray("c_id", data_slice->component_ids.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
        } 
    }

    static cudaError_t Compute_OutputLength(
        FrontierAttribute<SizeT> *frontier_attribute,
        SizeT       *d_offsets,
        VertexId    *d_indices,
        VertexId    *d_in_key_queue,
        SizeT       *partitioned_scanned_edges,
        SizeT        max_in,
        SizeT        max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false)
    {
        util::MemsetKernel<SizeT><<<1,1,0,stream>>>(frontier_attribute->output_length.GetPointer(util::DEVICE), frontier_attribute->queue_length, 1);
        return cudaSuccess;  
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
              char*           array)
    {
        Expand_Incoming_BothWay
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            keys_out,
            array_size,
            array);
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

        if (num_gpus < 2 && data_slice[0]->turn>0) return true;
        
        for (int gpu=0; gpu<num_gpus; gpu++)
            if (data_slice[gpu]->turn==0) return false;
        
        for (int gpu=0; gpu<num_gpus; gpu++)
        for (int peer=1; peer<num_gpus; peer++)
        for (int i=0; i<2; i++)
            if (data_slice[gpu]->in_length[i][peer]!=0) return false;

        for (int gpu=0; gpu<num_gpus; gpu++)
        for (int peer=0; peer<num_gpus; peer++)
            if (data_slice[gpu]->out_length[peer]!=0) return false;
        return true;
    }
};

    /**
     * @brief Enacts a connected component computing on the specified graph.
     *
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     * @tparam CCProblem CC Problem type.
     * @param[in] problem CCProblem object.
     * @param[in] max_grid_size Max grid size for CC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename CcEnactor>
    static CUT_THREADPROC CCThread(
        void * thread_data_)
    {
        typedef typename CcEnactor::Problem    Problem;
        typedef typename CcEnactor::SizeT      SizeT;
        typedef typename CcEnactor::VertexId   VertexId;
        typedef typename CcEnactor::Value      Value;
        typedef typename Problem::DataSlice    DataSlice;
        typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
        typedef UpdateMaskFunctor<VertexId, SizeT, Value, Problem> CcFunctor;
        ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
        Problem      *problem            =  (Problem*)     thread_data->problem;
        CcEnactor    *enactor            =  (CcEnactor*)   thread_data->enactor;
        int           num_gpus           =   problem     -> num_gpus;
        int           thread_num         =   thread_data -> thread_num;
        int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
        DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
        FrontierAttribute<SizeT>
                     *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
        EnactorStats *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

        data_slice -> work_progress      = &(enactor     -> work_progress      [thread_num * num_gpus]);
        do {
            printf("CCThread entered\n");fflush(stdout);
            if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
            thread_data->stats = 1;
            while (thread_data->stats !=2) sleep(0);
            thread_data->stats = 3;

            for (int peer_=0; peer_<num_gpus; peer_++)
            {
                frontier_attribute[peer_].queue_index  = 0;
                frontier_attribute[peer_].selector     = 0;
                frontier_attribute[peer_].queue_length = 0;
                frontier_attribute[peer_].queue_reset  = true;
                enactor_stats     [peer_].iteration    = 0;
            }
            if (num_gpus>1)
            {
                data_slice->vertex_associate_orgs[0]=data_slice->component_ids.GetPointer(util::DEVICE);
                data_slice->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
            }
            
            gunrock::app::Iteration_Loop
                <1,0, CcEnactor, CcFunctor, CCIteration<AdvanceKernelPolicy, FilterKernelPolicy, CcEnactor> > (thread_data);

            printf("CC_Thread finished\n");fflush(stdout);
        } while (0);
        thread_data->stats = 4;
        CUT_THREADEND;
    }


/**
 * @brief BC problem enactor class.
 *
 * @tparam INSTRUMENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <
    typename _Problem,
    bool _INSTRUMENT,                           // Whether or not to collect per-CTA clock-count statistics
    bool _DEBUG,
    bool _SIZE_CHECK>
class CCEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
    // Members
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

public:

    /**
     * @brief CCEnactor default constructor
     */
    CCEnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(EDGE_FRONTIERS, num_gpus, gpu_idx)
    {
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
    }

    /**
     * @brief CCEnactor default destructor
     */
    ~CCEnactor()
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
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        unsigned long long total_lifetimes = 0;
        unsigned long long total_runtimes  = 0;
        total_queued = 0;
        
        for (int gpu=0; gpu<this->num_gpus; gpu++)
        {
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

    template<
        typename AdvanceKernelPolity,
        typename FilterKernelPolicy>
    cudaError_t InitCC(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 512,
        bool        size_check    = true)
    {
        cudaError_t retval = cudaSuccess;
        //cpu_barrier = new util::cpu_mt::CPUBarrier[2];
        //cpu_barrier[0]=util::cpu_mt::CreateBarrier(this->num_gpus);
        //cpu_barrier[1]=util::cpu_mt::CreateBarrier(this->num_gpus);
        // Lazy initialization
        if (retval = EnactorBase <SizeT, DEBUG, SIZE_CHECK> ::Init(
            problem,
            max_grid_size,
            AdvanceKernelPolity::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        if (DEBUG) {
            printf("CC vertex map occupancy %d, level-grid size %d\n",
                        FilterKernelPolicy::CTA_OCCUPANCY, this->enactor_stats[0].filter_grid_size);
        }

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            //thread_slices[gpu].cpu_barrier  = cpu_barrier;
            thread_slices[gpu].thread_num   = gpu;
            thread_slices[gpu].problem      = (void*)problem;
            thread_slices[gpu].enactor      = (void*)this;
            thread_slices[gpu].context      =&(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats        = -1;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(CCThread<
                    AdvanceKernelPolity, FilterKernelPolicy,
                    CCEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
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
    cudaError_t EnactCC()
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
        if (this->DEBUG) printf("\nGPU CC Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Enact Kernel Entry, specify KernelPolicy
     *
     * @tparam CCProblem CC Problem type. @see CCProblem
     * @param[in] problem Pointer to CCProblem object.
     * @param[in] max_grid_size Max grid size for CC kernel calls. 
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename CCProblem>
    cudaError_t Enact()
        //CCProblem                      *problem,
        //int                             max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::advance::KernelPolicy<
                Problem,                            //Problem data type
                300,                                //CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY,
                7,                                  // LOG_THREADS,
                8,                                  // LOG_BLOCKS,
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE,
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS,
                32,                                 // WART_GATHER_THRESHOLD,
                128 * 4,                            // CTA_GATHER_THRESHOLD,
                7,                                  // LOG_SCHEDULE_GRANULARITY,
                gunrock::oprtr::advance::LB>
                    AdvancePolicy;

            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                0,                                  // END_BITMASK (no bitmask for cc)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterPolicy;

            return EnactCC<AdvancePolicy, FilterPolicy>();
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    cudaError_t Init(
            ContextPtr *context,
            Problem    *problem,
            int         max_grid_size = 512,
            bool        size_check    = true)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::advance::KernelPolicy<
                Problem,                            //Problem data type
                300,                                //CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY,
                7,                                  // LOG_THREADS,
                8,                                  // LOG_BLOCKS,
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE,
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS,
                32,                                 // WART_GATHER_THRESHOLD,
                128 * 4,                            // CTA_GATHER_THRESHOLD,
                7,                                  // LOG_SCHEDULE_GRANULARITY,
                gunrock::oprtr::advance::LB>
                    AdvancePolicy;

            typedef gunrock::oprtr::filter::KernelPolicy<
                Problem,                            // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                0,                                  // END_BITMASK (no bitmask for cc)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterPolicy;

            return InitCC<AdvancePolicy, FilterPolicy>(
                    context, problem, max_grid_size, size_check);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }
    /** @} */
};

} // namespace cc
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
