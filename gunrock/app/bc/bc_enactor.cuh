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
#include <gunrock/util/device_intrinsics.cuh>

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

/*
 * @brief Expand incoming function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] num_elements
 * @param[in] keys_in
 * @param[in] keys_out
 * @param[in] array_size
 * @param[in] array
 */
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
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
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

    x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

    while (x<num_elements)
    {
        VertexId key = keys_in[x];

        if (atomicCAS(s_vertex_associate_org[1] + key, (VertexId)-2, 
            s_vertex_associate_in[1][x]) == -2)
        {
            s_vertex_associate_org[0][key] = s_vertex_associate_in[0][x];
            t = -1;
        } else {
            t = atomicCAS(s_vertex_associate_org[0] + key, (VertexId)-1, 
                s_vertex_associate_in[0][x]);
            if (s_vertex_associate_org[0][key] != s_vertex_associate_in[0][x])
            {
                keys_out[x] = -1;
                x += STRIDE;
                continue;
            }
        }
        if (t == -1) keys_out[x]=key; 
        else keys_out[x]=-1;
        atomicAdd(s_value__associate_org[0] + key, s_value__associate_in[0][x]);
        x += STRIDE;
    }
}

template <
    typename KernelPolicy>
__global__ void Expand_Incoming_Forward_Kernel(
    int                              thread_num,
    typename KernelPolicy::SizeT     num_elements,
    typename KernelPolicy::VertexId  label,
    typename KernelPolicy::VertexId *d_keys_in,
    typename KernelPolicy::VertexId *d_vertex_associate_in,
    typename KernelPolicy::Value    *d_value__associate_in,
    typename KernelPolicy::SizeT    *d_out_length,
    typename KernelPolicy::VertexId *d_keys_out,
    typename KernelPolicy::VertexId *d_labels,
    typename KernelPolicy::VertexId *d_preds,
    typename KernelPolicy::Value    *d_sigmas)
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Problem::MaskT    MaskT;
    typedef typename KernelPolicy::Value    Value;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

    while (x - threadIdx.x < num_elements)
    {
        bool to_process = true;
        VertexId key = util::InvalidValue<VertexId>();

        if (x < num_elements)
        {
            key = d_keys_in[x];
            //VertexId label_in = d_vertex_associate_in[x << 1];
            //VertexId pred_in  = d_vertex_associate_in[(x << 1) + 1];
            //VertexId pred_in = d_vertex_associate_in[x];
            //VertexId org_label = __ldg(d_labels + key);
            VertexId org_label = atomicCAS(d_labels + key, (VertexId)-1, label);
            //if (atomicCAS(d_preds + key, (VertexId)-2, pred_in) == -2)
            //if (d_preds[key] == -2)
            if (org_label == -1)
            {
                //d_preds[key] = d_vertex_associate_in[x];
                //d_preds[key] = pred_in;
                //d_labels[key] = label;
                //atomicAdd(d_sigmas + key, d_value__associate_in[x]);
            } else {
                //VertexId org_label = d_labels[key];
                //if (org_label == label || org_label == -1)
                //    atomicAdd(d_sigmas + key, d_value__associate_in[x]);
                to_process = false;
            }
            if (org_label == label || org_label == -1)
                atomicAdd(d_sigmas + key, d_value__associate_in[x]);
        } else to_process = false;

        SizeT output_pos = util::InvalidValue<SizeT>();
        BlockScanT::LogicScan(to_process, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            if (output_pos != 0 || to_process)
            block_offset = atomicAdd(d_out_length, output_pos + ((to_process) ? 1 : 0));
        }
        __syncthreads();

        if (to_process)
        {
            output_pos += block_offset;
            d_keys_out[output_pos] = key;
        }
        x += STRIDE;
    }
}

/*
 * @brief Expand incoming function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] num_elements
 * @param[in] keys_in
 * @param[in] keys_out
 * @param[in] array_size
 * @param[in] array
 */
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
    const SizeT STRIDE = (SizeT) gridDim.x * blockDim.x;
    size_t      offset                = 0;
    offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_in  = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
    offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_org = (Value**   )&(s_array[offset]);
    SizeT x = threadIdx.x;
    if (x < array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;

    while (x<num_elements)
    {
        VertexId key=keys_in[x];
        keys_out[x]=key;
        s_value__associate_org[0][key] = s_value__associate_in[0][x];
        s_value__associate_org[1][key] = s_value__associate_in[1][x];
        x += STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT,
    typename Value>
__global__ void Expand_Incoming_Backward_Kernel (
    const SizeT           num_elements,
    const VertexId* const d_keys_in,
    const Value*    const d_values_in,
          SizeT           output_offset,
          VertexId*       d_keys_out,
          Value*          d_deltas,
          Value*          d_bc_values,
          VertexId*       d_original_vertex)
{
    SizeT x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) gridDim.x * blockDim.x;
    while (x < num_elements)
    {
        VertexId key = d_keys_in[x];
        d_deltas   [key] = d_values_in[x << 1];
        d_bc_values[key] = d_values_in[(x << 1) + 1];
        //d_keys_out[x + output_offset] = key;
        //printf("Expand_Incoming : [%2d, %2d] -> %.4f, %.4f\n",
        //    x, d_original_vertex[key], d_values_in[x << 1], d_values_in[(x<<1)+1]);
        x += STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT,
    typename Value>
__global__ void Assign_Middle(
    int        num_gpus,
    SizeT      num_elements,
    VertexId  *d_keys_in,
    VertexId  *d_labels_in,
    Value     *d_sigmas_in,
    //VertexId **d_keys_outs,
    VertexId  *d_labels_out,
    Value     *d_sigmas_out)
{
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) gridDim.x * blockDim.x;
    /*__shared__ VertexId* s_keys_outs[8];
    __shared__ VertexId* s_labels_outs[8];
    __shared__ Value*    s_sigmas_outs[8];
    if (threadIdx.x < num_gpus)
    {
        s_keys_outs  [threadIdx.x] = d_keys_outs  [threadIdx.x];
        s_labels_outs[threadIdx.x] = d_labels_outs[threadIdx.x];
        s_sigmas_outs[threadIdx.x] = d_sigmas_outs[threadIdx.x];
    }
    __syncthreads();*/

    while (x < num_elements)
    {
        VertexId key = d_keys_in[x];
        /*VertexId label = d_labels_in[key];
        Value    sigma = d_sigmas_in[key];
        for (int gpu = 1; gpu < num_gpus; gpu++)
        {
            s_keys_outs  [gpu][x] = key;
            s_labels_outs[gpu][x] = label;
            s_sigmas_outs[gpu][x] = sigma;
        }*/
        d_labels_out[x] = d_labels_in[key];
        d_sigmas_out[x] = d_sigmas_in[key];
        
        x += STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT,
    typename Value>
__global__ void Expand_Incoming_Middle(
    SizeT num_elements,
    VertexId* d_keys_in,
    VertexId* d_labels_in,
    Value*    d_sigmas_in,
    VertexId* d_labels_out,
    Value*    d_sigmas_out)
{
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) gridDim.x * blockDim.x;
    while (x < num_elements)
    {
        VertexId key = d_keys_in[x];
        d_labels_out[key] = d_labels_in[x];
        d_sigmas_out[key] = d_sigmas_in[x];
       
        x += STRIDE;
    }
}

/*
 * @brief Iteration structure derived from IterationBase.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor we process on.
 */
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
    typedef GraphSlice<VertexId, SizeT, Value> 
                                         GraphSliceT;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value>
                                         Frontier  ;
    typedef ForwardFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> ForwardFunctor;
    typedef IterationBase <
        AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        false, true, false, true, true>  BaseIteration;

    /*
     * @brief FullQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Gather(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (enactor_stats->iteration <= 0) return;

        SizeT cur_offset = data_slice->forward_queue_offsets[peer_].back();
        bool oversized = false;
        if (enactor_stats->retval =
            Check_Size<SizeT, VertexId> (
                enactor -> size_check, "forward_output", 
                cur_offset + frontier_attribute -> queue_length, 
                &data_slice -> forward_output[peer_], 
                oversized, thread_num, enactor_stats -> iteration, peer_)) return;
        util::MemsetCopyVectorKernel<<<120, 512, 0, stream>>>(
            data_slice -> forward_output[peer_].GetPointer(util::DEVICE) + cur_offset,
            frontier_queue -> keys[ frontier_attribute -> selector].GetPointer(util::DEVICE),
            frontier_attribute -> queue_length);
        data_slice -> forward_queue_offsets[peer_].push_back(
            frontier_attribute -> queue_length+cur_offset);
    }

    /*
     * @brief FullQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Core(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        frontier_attribute-> queue_reset = true;
        enactor_stats     -> nodes_queued[0] += frontier_attribute->queue_length;

        if (enactor -> debug) 
            util::cpu_mt::PrintMessage("Advance begin",
                thread_num, enactor_stats->iteration, peer_);
        // Edge Map
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, ForwardFunctor, gunrock::oprtr::advance::V2V>(
            enactor_stats[0],
            frontier_attribute[0],
            enactor_stats -> iteration + 1,
            data_slice,
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges ->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),// d_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
            (Value*   )NULL,
            (Value*   )NULL,
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes, //graph_slice->frontier_elements[frontier_attribute.selector],  // max_in_queue
            graph_slice->edges, //graph_slice->frontier_elements[frontier_attribute.selector^1],// max_out_queue
            work_progress[0],
            context[0],
            stream,
            //gunrock::oprtr::advance::V2V,
            false,
            false,
            false);
        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Advance end",
                thread_num, enactor_stats->iteration, peer_);

        frontier_attribute -> queue_reset = false;
        frontier_attribute -> queue_index++;
        frontier_attribute -> selector ^= 1;
        if (gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>())
        {
            enactor_stats -> edges_queued[0] += frontier_attribute -> output_length[0];
        } else {
            enactor_stats      -> AccumulateEdges(
                work_progress  -> template GetQueueLengthPointer<unsigned int>(
                    frontier_attribute -> queue_index), stream);
        }

        if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
        {
            if (enactor -> debug) 
                util::cpu_mt::PrintMessage("Filter begin", 
                    thread_num, enactor_stats->iteration, peer_);
            // Filter
            /*gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, Problem, ForwardFunctor>(
                enactor_stats->filter_grid_size, 
                FilterKernelPolicy::THREADS, 
                (size_t)0, 
                stream,
                enactor_stats->iteration+1,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),// d_in_queue
                (Value*) NULL,
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                d_data_slice,
                (unsigned char*)NULL,
                work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),// max_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(),// max_out_queue
                enactor_stats->filter_kernel_stats);*/
            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, Problem, ForwardFunctor>(
                enactor_stats[0],
                frontier_attribute[0],
                (VertexId)enactor_stats -> iteration + 1,
                data_slice,
                d_data_slice,
                (SizeT*)NULL,
                (unsigned char*)NULL,
                frontier_queue -> keys[frontier_attribute -> selector  ].GetPointer(util::DEVICE),
                frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE),
                (Value*)NULL,
                (Value*)NULL,
                util::InvalidValue<SizeT>(),
                graph_slice -> nodes,
                work_progress[0],
                context[0],
                stream,
                frontier_queue -> keys[frontier_attribute -> selector  ].GetSize(),
                frontier_queue -> keys[frontier_attribute -> selector^1].GetSize(),
                enactor_stats -> filter_kernel_stats);

            if (enactor -> debug && (enactor_stats->retval = 
                util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
            if (enactor -> debug) 
                util::cpu_mt::PrintMessage("Filter end.", 
                    thread_num, enactor_stats->iteration, peer_);

            frontier_attribute -> queue_index++;
            frontier_attribute -> selector ^= 1;
        }
        if (enactor_stats  -> retval = 
            work_progress  -> GetQueueLength(
                frontier_attribute -> queue_index, 
                frontier_attribute -> queue_length,
                false,stream,true)) 
            return;
        //if (enactor_stats->retval = util::GRError(
        //    cudaStreamSynchronize(stream), 
        //   "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
    }

    /*
     * @brief Expand incoming function.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] grid_size
     * @param[in] block_size
     * @param[in] shared_size
     * @param[in] stream
     * @param[in] num_elements
     * @param[in] keys_in
     * @param[in] keys_out
     * @param[in] array_size
     * @param[in] array
     * @param[in] data_slice
     */
    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming_Old(
              Enactor        *enactor,
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

        Check_Size<SizeT, VertexId>(
            enactor -> size_check, "queue1", 
            num_elements, keys_out, over_sized, -1, -1, -1);
        Expand_Incoming_Forward
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            keys_out->GetPointer(util::DEVICE),
            array_size,
            array);
    }

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
        Enactor        *enactor,
        cudaStream_t    stream,
        VertexId        iteration,
        int             peer_,
        SizeT           received_length,
        SizeT           num_elements,
        util::Array1D<SizeT, SizeT    > &out_length,
        util::Array1D<SizeT, VertexId > &keys_in,
        util::Array1D<SizeT, VertexId > &vertex_associate_in,
        util::Array1D<SizeT, Value    > &value__associate_in,
        util::Array1D<SizeT, VertexId > &keys_out,
        util::Array1D<SizeT, VertexId*> &vertex_associate_orgs,
        util::Array1D<SizeT, Value   *> &value__associate_orgs,
        DataSlice      *h_data_slice,
        EnactorStats<SizeT> *enactor_stats)
    {
        bool over_sized = false;
        if (enactor -> problem -> unified_receive)
        {    
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "incoming_queue",
                num_elements + received_length,
                &keys_out, over_sized, h_data_slice -> gpu_idx, iteration, peer_, true))
                return;
            received_length += num_elements;
        } else {
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "incomping_queue",
                num_elements,
                &keys_out, over_sized, h_data_slice -> gpu_idx, iteration, peer_))
                return;
            out_length[peer_] =0;
            out_length.Move(util::HOST, util::DEVICE, 1, peer_, stream);
        }

        int num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2+ 1;
        if (num_blocks > 120) num_blocks = 120; 
        Expand_Incoming_Forward_Kernel
            <AdvanceKernelPolicy>
            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
            (h_data_slice -> gpu_idx,
            num_elements,
            iteration,
            keys_in.GetPointer(util::DEVICE),
            vertex_associate_in.GetPointer(util::DEVICE),
            value__associate_in.GetPointer(util::DEVICE),
            out_length.GetPointer(util::DEVICE) + ((enactor -> problem -> unified_receive) ? 0: peer_),
            keys_out.GetPointer(util::DEVICE),
            h_data_slice -> labels.GetPointer(util::DEVICE),
            h_data_slice -> preds .GetPointer(util::DEVICE),
            h_data_slice -> sigmas.GetPointer(util::DEVICE));

        if (!enactor -> problem -> unified_receive)
            out_length.Move(util::DEVICE, util::HOST, 1, peer_, stream);
        else out_length.Move(util::DEVICE, util::HOST, 1, 0, stream); 
    }

    /*
     * @brief Compute output queue length function.
     *
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] d_offsets Pointer to the offsets.
     * @param[in] d_indices Pointer to the indices.
     * @param[in] d_in_key_queue Pointer to the input mapping queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] max_in Maximum input queue size.
     * @param[in] max_out Maximum output queue size.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     * @param[in] ADVANCE_TYPE Advance kernel mode.
     * @param[in] express Whether or not enable express mode.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    static cudaError_t Compute_OutputLength(
        Enactor                        *enactor,
        FrontierAttribute<SizeT>       *frontier_attribute,
        SizeT                          *d_offsets,
        VertexId                       *d_indices,
        SizeT                          *d_inv_offsets,
        VertexId                       *d_inv_indices,
        VertexId                       *d_in_key_queue,
        util::Array1D<SizeT, SizeT>    *partitioned_scanned_edges,
        SizeT                          max_in,
        SizeT                          max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false,
        bool                            in_inv = false,
        bool                           out_inv = false)
    {
        cudaError_t retval = cudaSuccess;
        bool over_sized = false;

        if (!enactor -> size_check &&
            (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
        {    
            frontier_attribute -> output_length[0] = 0; 
            return retval;

        } else {
            if (retval = Check_Size<SizeT, SizeT> (
                enactor -> size_check, "scanned_edges", 
                frontier_attribute -> queue_length + 2, 
                partitioned_scanned_edges, over_sized, -1, -1, -1, false)) 
                return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, ForwardFunctor, gunrock::oprtr::advance::V2V>(
                frontier_attribute,
                d_offsets,
                d_indices,
                d_inv_offsets,
                d_inv_indices,
                d_in_key_queue,
                partitioned_scanned_edges -> GetPointer(util::DEVICE),
                max_in,
                max_out,
                context,
                stream,
                //ADVANCE_TYPE,
                express,
                in_inv,
                out_inv);
            frontier_attribute -> output_length.Move(
                util::DEVICE, util::HOST, 1, 0, stream);
        }
        return retval;
    }

    /*
     * @brief Check frontier queue size function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] request_length Request frontier queue length.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     */
    static void Check_Queue_Size(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        Frontier                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        GraphSliceT                   *graph_slice)
    {
        bool over_sized = false;
        int  selector   = frontier_attribute -> selector;
        int  iteration  = enactor_stats -> iteration;

        if (enactor -> debug)
        {
            printf("%d\t %d\t %d\t queue_length = %lld, output_length = %lld, @ %d\n",
                thread_num, iteration, peer_,
                (long long)frontier_queue->keys[selector^1].GetSize(),
                (long long)request_length, selector);
            fflush(stdout);
        }

        if (!enactor -> size_check &&
            (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
        {
            frontier_attribute -> output_length[0] = 0;
            return;
        } else if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
        {
            if (enactor_stats -> retval =
                Check_Size<SizeT, VertexId > (
                    true, "queue3", request_length, 
                    &frontier_queue -> keys  [selector^1], 
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats -> retval =
                Check_Size<SizeT, VertexId > (
                    true, "queue3", graph_slice->nodes+2, 
                    &frontier_queue -> keys  [selector  ], 
                    over_sized, thread_num, iteration, peer_, true )) return;
            if (enactor -> problem -> use_double_buffer)
            {
                if (enactor_stats -> retval =
                    Check_Size<SizeT, Value> (
                        true, "queue3", request_length, 
                        &frontier_queue -> values[selector^1], 
                        over_sized, thread_num, iteration, peer_, false)) return;
                if (enactor_stats -> retval =
                    Check_Size<SizeT, Value> (
                        true, "queue3", graph_slice -> nodes+2, 
                        &frontier_queue -> values[selector  ], 
                        over_sized, thread_num, iteration, peer_, true )) return;
            }
        } else {
            if (enactor_stats->retval =
                Check_Size< SizeT, VertexId > (
                    true, "queue3", graph_slice -> nodes * 1.2,
                    &frontier_queue->keys  [selector^1],
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor -> problem -> use_double_buffer)
            {
                if (enactor_stats->retval =
                    Check_Size< SizeT, Value> (
                        true, "queue3", graph_slice->nodes * 1.2,
                        &frontier_queue->values[selector^1],
                        over_sized, thread_num, iteration, peer_, false )) return;
            }
        }
    }

    /*   
     * @brief Iteration_Update_Preds function.
     *
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] num_elements Number of elements.
     * @param[in] stream CUDA stream.
     */
    static void Iteration_Update_Preds(
        Enactor                       *enactor,
        GraphSliceT                   *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT>
                                      *frontier_attribute,
        Frontier                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {    
        return ;
    }    
};

/*
 * @brief Iteration structure derived from IterationBase.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor we process on.
 */
template <
    typename AdvanceKernelPolicy, 
    typename FilterKernelPolicy, 
    typename Enactor>
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
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<VertexId, SizeT, Value> 
                                         GraphSliceT;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value>
                                         Frontier  ;
    typedef IterationBase <
        AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        false, true, true, false, false> BaseIteration;
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

    /*
     * @brief FullQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Gather(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        SizeT cur_pos = data_slice -> forward_queue_offsets[peer_].back();
        data_slice -> forward_queue_offsets[peer_].pop_back();
        SizeT pre_pos = data_slice -> forward_queue_offsets[peer_].back();
        frontier_attribute -> queue_reset  = true;
        frontier_attribute -> selector     = 0;//frontier_queue->keys[0].GetSize() > frontier_queue->keys[1].GetSize() ? 1 : 0;
        if (enactor_stats -> iteration>0 && cur_pos - pre_pos >0)
        {
            frontier_attribute -> queue_length = cur_pos - pre_pos;
            bool over_sized = false;
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId> (
                enactor -> size_check, "queue1", 
                frontier_attribute -> queue_length, 
                &frontier_queue -> keys[frontier_queue -> selector], 
                over_sized, thread_num, enactor_stats->iteration, peer_, false)) return;
            util::MemsetCopyVectorKernel<<<120, 512, 0, stream>>>(
                frontier_queue -> keys[frontier_queue -> selector].GetPointer(util::DEVICE),
                data_slice ->forward_output[peer_].GetPointer(util::DEVICE) + pre_pos,
                frontier_attribute -> queue_length);
        }
        else frontier_attribute -> queue_length = 0;
    }

    /*
     * @brief FullQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Core(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
       /*frontier_attribute.queue_length        = graph_slice->nodes;
        // Fill in the frontier_queues
        util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_keys[0], graph_slice->nodes);
        // Filter
        gunrock::oprtr::filter::LaunchKernel
            <FilterKernelPolicy, BCProblem, BackwardFunctor>(
            enactor_stats.filter_grid_size,
            FilterKernelPolicy::THREADS,
            0, 0,
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
        if (enactor_stats->iteration > 0) 
        {
            //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pre advance",
            //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
            //    frontier_attribute -> queue_length, thread_num, enactor_stats -> iteration, -1, stream);
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, BackwardFunctor, gunrock::oprtr::advance::V2V>(
                enactor_stats[0],
                frontier_attribute[0],
                enactor_stats -> iteration + 1,
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges ->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                (VertexId*)NULL, //frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                (Value*   )NULL,
                (Value*   )NULL,
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,                 // max_in_queue
                graph_slice->edges,                 // max_out_queue
                work_progress[0],
                context[0],
                stream,
                //gunrock::oprtr::advance::V2V,
                false,
                false,
                false);
        } else {
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, BackwardFunctor2, gunrock::oprtr::advance::V2V>(
                enactor_stats[0],
                frontier_attribute[0],
                enactor_stats -> iteration + 1,
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges ->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                (VertexId*)NULL, //frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                (Value*   )NULL,
                (Value*   )NULL,
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,                 // max_in_queue
                graph_slice->edges,                 // max_out_queue
                work_progress[0],
                context[0],
                stream,
                //gunrock::oprtr::advance::V2V,
                false,
                false,
                false);
        }
        enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
        /*frontier_attribute -> queue_index++;
        //frontier_attribute -> selector ^= 1;
        if (enactor_stats  -> retval = 
            work_progress  -> GetQueueLength(
                frontier_attribute -> queue_index, 
                frontier_attribute -> queue_length,
                false,stream,true)) 
            return;
        if (enactor_stats->retval = util::GRError(
            cudaStreamSynchronize(stream), 
           "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
        */
    }

    /*
     * @brief Expand incoming function.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] grid_size
     * @param[in] block_size
     * @param[in] shared_size
     * @param[in] stream
     * @param[in] num_elements
     * @param[in] keys_in
     * @param[in] keys_out
     * @param[in] array_size
     * @param[in] array
     * @param[in] data_slice
     *
     */
    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming_Old(
        Enactor        *enactor,
        int             grid_size,
        int             block_size,
        size_t          shared_size,
        cudaStream_t    stream,
        SizeT           &num_elements,
        VertexId*       keys_in,
        util::Array1D<SizeT, VertexId>* keys_out,
        const size_t    array_size,
        char*           array,
        DataSlice*      data_slice)
    {
        bool over_sized = false;
        Check_Size<SizeT, VertexId>(
            enactor -> size_check, "queue1", 
            num_elements, keys_out, over_sized, -1, -1, -1);
        Expand_Incoming_Backward
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            keys_out->GetPointer(util::DEVICE),
            array_size,
            array);
    }

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
        Enactor        *enactor,
        cudaStream_t    stream,
        VertexId        iteration,
        int             peer_,
        SizeT           received_length,
        SizeT           num_elements,
        util::Array1D<SizeT, SizeT    > &out_length,
        util::Array1D<SizeT, VertexId > &keys_in,
        util::Array1D<SizeT, VertexId > &vertex_associate_in,
        util::Array1D<SizeT, Value    > &value__associate_in,
        util::Array1D<SizeT, VertexId > &keys_out,
        util::Array1D<SizeT, VertexId*> &vertex_associate_orgs,
        util::Array1D<SizeT, Value   *> &value__associate_orgs,
        DataSlice      *h_data_slice,
        EnactorStats<SizeT> *enactor_stats)
    {
        bool over_sized = false;
        if (h_data_slice -> first_backward_incoming[peer_])
        {
            h_data_slice -> first_backward_incoming[peer_] = false;
            return;
        }
        if (enactor -> problem -> unified_receive)
        {    
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "incoming_queue",
                num_elements + received_length,
                &keys_out, over_sized, h_data_slice -> gpu_idx, iteration, peer_, true))
                return;
        } else {
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "incomping_queue",
                num_elements,
                &keys_out, over_sized, h_data_slice -> gpu_idx, iteration, peer_))
                return;
            //out_length[peer_] =0;
            //out_length.Move(util::HOST, util::DEVICE, 1, peer_, stream);
        }

        int num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2+ 1;
        if (num_blocks > 120) num_blocks = 120; 
        Expand_Incoming_Backward_Kernel
            <VertexId, SizeT, Value>
            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
            (num_elements,
            keys_in.GetPointer(util::DEVICE),
            value__associate_in.GetPointer(util::DEVICE),
            (enactor -> problem -> unified_receive) ? received_length : 0, //out_length.GetPointer(util::DEVICE) + ((enactor -> problem -> unified_receive) ? 0: peer_),
            keys_out.GetPointer(util::DEVICE),
            h_data_slice -> deltas   .GetPointer(util::DEVICE),
            h_data_slice -> bc_values.GetPointer(util::DEVICE),
            h_data_slice -> original_vertex.GetPointer(util::DEVICE));

        if (!enactor -> problem -> unified_receive)
        {
            //out_length.Move(util::DEVICE, util::HOST, 1, peer_, stream);
            out_length[peer_] = num_elements;
        } else {
            //out_length.Move(util::DEVICE, util::HOST, 1, 0, stream); 
            received_length += num_elements;
            out_length[0] = received_length;
        }
    }
    
    /*
     * @brief Compute output queue length function.
     *
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] d_offsets Pointer to the offsets.
     * @param[in] d_indices Pointer to the indices.
     * @param[in] d_in_key_queue Pointer to the input mapping queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] max_in Maximum input queue size.
     * @param[in] max_out Maximum output queue size.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     * @param[in] ADVANCE_TYPE Advance kernel mode.
     * @param[in] express Whether or not enable express mode.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    static cudaError_t Compute_OutputLength(
        Enactor                        *enactor,
        FrontierAttribute<SizeT>       *frontier_attribute,
        SizeT                          *d_offsets,
        VertexId                       *d_indices,
        SizeT                          *d_inv_offsets,
        VertexId                       *d_inv_indices,
        VertexId                       *d_in_key_queue,
        util::Array1D<SizeT, SizeT>    *partitioned_scanned_edges,
        SizeT                          max_in,
        SizeT                          max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false,
        bool                           in_inv = false,
        bool                           out_inv = false)
    {
        cudaError_t retval = cudaSuccess;
        bool over_sized = false;

        if (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>())
        {    
            frontier_attribute -> output_length[0] = 0; 
            return retval;

        } else {
            if (retval = Check_Size<SizeT, SizeT> (
                enactor -> size_check, "scanned_edges", 
                frontier_attribute->queue_length + 2, 
                partitioned_scanned_edges, 
                over_sized, -1, -1, -1, false)) return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, BackwardFunctor, gunrock::oprtr::advance::V2V>(
                frontier_attribute,
                d_offsets,
                d_indices,
                d_inv_offsets,
                d_inv_indices,
                d_in_key_queue,
                partitioned_scanned_edges->GetPointer(util::DEVICE),
                max_in,
                max_out,
                context,
                stream,
                //ADVANCE_TYPE,
                express,
                in_inv,
                out_inv);
            frontier_attribute -> output_length.Move(
                util::DEVICE, util::HOST, 1, 0, stream);
        }
        return retval;
    }

    static void Iteration_Change(long long &iterations)
    {
        iterations--;
    }

    /*
     * @brief Stop_Condition check function.
     *
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] num_gpus Number of GPUs used.
     */
    static bool Stop_Condition(
        EnactorStats<SizeT>             *enactor_stats,
        FrontierAttribute<SizeT>        *frontier_attribute,
        util::Array1D<SizeT, DataSlice> *data_slice,
        int num_gpus)
    {
        for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++)
        if (enactor_stats[gpu].retval!=cudaSuccess)
        {
            printf("(CUDA error %d @ GPU %d: %s\n", 
                enactor_stats[gpu].retval, gpu, 
                cudaGetErrorString(enactor_stats[gpu].retval)); 
            fflush(stdout);
            return true;
        }
        
        if (All_Done(enactor_stats, frontier_attribute, data_slice, num_gpus))
        {
            for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
            if (enactor_stats[gpu].iteration>1) 
            {
                //printf("enactor_stats[%d].iteration = %d\n",
                //    gpu, enactor_stats[gpu].iteration);
                return false;
            }
            return true;
        } else {
            //bool has_negetive = false;
            for (int gpu=0; gpu<num_gpus * num_gpus; gpu++)
            if (enactor_stats[gpu].iteration < 0)
            {
                return true;
            }
            return false;
        }
    }

    /*
     * @brief Check frontier queue size function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] request_length Request frontier queue length.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     */
    static void Check_Queue_Size(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        Frontier                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        GraphSliceT                   *graph_slice)
    {
        return; // no need to check for backward
        /*bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (enactor -> debug)
        {
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                thread_num, iteration, peer_,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);
            fflush(stdout);
        }

        if (enactor_stats->retval =
            Check_Size<SizeT, VertexId > (
                true, "queue3", request_length, 
                &frontier_queue->keys  [selector^1], 
                over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval =
            Check_Size<SizeT, VertexId > (
                true, "queue3", graph_slice->nodes+2, 
                &frontier_queue->keys  [selector  ], 
                over_sized, thread_num, iteration, peer_, true )) return;
        if (enactor -> problem -> use_double_buffer)
        {
            if (enactor_stats->retval =
                Check_Size<SizeT, Value> (
                    true, "queue3", request_length, 
                    &frontier_queue->values[selector^1], 
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size<SizeT, Value> (
                    true, "queue3", graph_slice->nodes+2, 
                    &frontier_queue->values[selector  ], 
                    over_sized, thread_num, iteration, peer_, true )) return;
        }*/
    }

    /*   
     * @brief Iteration_Update_Preds function.
     *
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] num_elements Number of elements.
     * @param[in] stream CUDA stream.
     */
    static void Iteration_Update_Preds(
        Enactor                       *enactor,
        GraphSliceT                   *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT>
                                      *frontier_attribute,
        Frontier                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {    
        return ;
    }    
};

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam BcEnactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
static CUT_THREADPROC BCThread(
    void * thread_data_)
{
    typedef typename Enactor::Problem    Problem;
    typedef typename Enactor::SizeT      SizeT;
    typedef typename Enactor::VertexId   VertexId;
    typedef typename Enactor::Value      Value;
    typedef typename Problem::DataSlice  DataSlice;
    typedef GraphSlice     <VertexId, SizeT, Value>          GraphSliceT;
    typedef ForwardFunctor <VertexId, SizeT, Value, Problem> BcFFunctor;
    typedef BackwardFunctor<VertexId, SizeT, Value, Problem> BcBFunctor;

    ThreadSlice  *thread_data         =  (ThreadSlice*) thread_data_;
    Problem      *problem             =  (Problem*)     thread_data->problem;
    Enactor      *enactor             =  (Enactor*)     thread_data->enactor;
    //util::cpu_mt::CPUBarrier
    //             *cpu_barrier         =   thread_data->cpu_barrier;
    int           num_gpus            =   problem     -> num_gpus;
    int           thread_num          =   thread_data -> thread_num;
    int           gpu_idx             =   problem     -> gpu_idx            [thread_num] ;
    DataSlice    *data_slice          =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice        =   problem     -> data_slices;
    //GraphSliceT  *graph_slice         =   problem     -> graph_slices       [thread_num] ;
    FrontierAttribute<SizeT>
                 *frontier_attribute  = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    EnactorStats<SizeT> *enactor_stats       = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
    EnactorStats<SizeT> *s_enactor_stats     = &(enactor     -> enactor_stats      [0                    ]);
    bool          has_error           = false;
    //util::Array1D<int, unsigned char>* barrier_markers = data_slice -> barrier_markers;

    if (enactor_stats[0].retval = util::SetDevice(gpu_idx))
    {
        thread_data -> status = ThreadSlice::Status::Ended;
        CUT_THREADEND;
    }

    if (enactor -> debug)
        util::cpu_mt::PrintMessage("Thread started",
            thread_num);

    thread_data->status = ThreadSlice::Status::Idle;
    while (thread_data -> status != ThreadSlice::Status::ToKill)
    {
        while (thread_data -> status == ThreadSlice::Status::Wait ||
               thread_data -> status == ThreadSlice::Status::Idle)
        {
            sleep(0);
            //std::this_thread::yield();
        }
        if (thread_data -> status == ThreadSlice::Status::ToKill)
            break;
        //thread_data->status = ThreadSlice::Status::Running;

        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Enact begins",
                thread_num);

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
            //data_slice->vertex_associate_orgs[0]=data_slice->labels.GetPointer(util::DEVICE);
            //data_slice->vertex_associate_orgs[0]=data_slice->preds.GetPointer(util::DEVICE);
            data_slice->value__associate_orgs[0]=data_slice->sigmas.GetPointer(util::DEVICE);
            //data_slice->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
            data_slice->value__associate_orgs.Move(util::HOST, util::DEVICE);
        }
        gunrock::app::Iteration_Loop
            <Enactor, BcFFunctor, 
            Forward_Iteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>,
            0, 1 > (thread_data);
        if (enactor -> debug) 
            util::cpu_mt::PrintMessage("Forward phase finished.", 
                thread_num, enactor_stats->iteration);

        if (num_gpus>1)
        {
            data_slice -> middle_iteration = enactor_stats -> iteration;
            int middle_event_markers[8];
            int middle_event_counter = 0;
            /*for (int gpu = 0; gpu < num_gpus; gpu++)
                middle_event_markers[gpu] = 0;
            while (middle_event_counter < num_gpus)
            {
                for (int gpu = 0; gpu < num_gpus; gpu++)
                {
                    if (middle_event_markers[gpu] == 1) continue;
                    if (s_data_slice[gpu] -> middle_iteration != 0)
                    {
                        middle_event_markers[gpu] = 1;
                        middle_event_counter ++;
                    }
                }
                if (middle_event_counter < num_gpus)
                    sleep(0); 
            }*/
            has_error =false;
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                while (s_data_slice[gpu] -> middle_iteration < 0 && !has_error)
                {
                    sleep(0); 
                    for (int i=0; i<num_gpus * num_gpus; i++)
                    {
                        if (s_enactor_stats[i].retval != cudaSuccess)
                        {
                            has_error = true;
                            break;
                        }
                    }
                }
                if (has_error) break;
            }
            if (has_error)
            {
                thread_data -> status = ThreadSlice::Status::Idle;
                continue;
            }
            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Barrier1 past",
                    thread_num, enactor_stats -> iteration); 

            int num_blocks = data_slice -> local_vertices.GetSize() / AdvanceKernelPolicy::THREADS / 2 +1;
            if (num_blocks > 480) num_blocks = 480;
            Assign_Middle<VertexId, SizeT, Value>
                <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, data_slice -> streams[0]>>>(
                num_gpus,
                data_slice -> local_vertices.GetSize(),
                data_slice -> local_vertices.GetPointer(util::DEVICE),
                data_slice -> labels.GetPointer(util::DEVICE),
                data_slice -> sigmas.GetPointer(util::DEVICE),
                //data_slice -> keys_outs.GetPointer(util::DEVICE),
                data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE),
                data_slice -> value__associate_out[1].GetPointer(util::DEVICE));
            for (int peer_ = 1; peer_ < num_gpus; peer_ ++)
            {
                data_slice -> keys_out            [peer_].ForceSetPointer(
                    data_slice -> local_vertices.GetPointer(util::DEVICE), util::DEVICE);
                data_slice -> vertex_associate_out[peer_].ForceSetPointer(
                    data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE), util::DEVICE);
                data_slice -> value__associate_out[peer_].ForceSetPointer(
                    data_slice -> value__associate_out[1].GetPointer(util::DEVICE), util::DEVICE);
            }
            if (enactor_stats -> retval = util::GRError(cudaEventRecord(
                data_slice -> middle_events[0], data_slice -> streams[0]),
                "cudaEventRecord failed", __FILE__, __LINE__))
                //break;
            {
                thread_data -> status = ThreadSlice::Status::Idle;
                continue;
            }

            //if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(data_slice -> streams[0]),
            //    "cudaStreamSynchronzie failed", __FILE__, __LINE__)) break;
            for (int peer = 0; peer < num_gpus; peer++)
            {
                if (peer == thread_num) continue;
                int peer_ = peer < thread_num ? peer + 1 : peer;
                if (enactor_stats -> retval = util::GRError(
                    cudaStreamWaitEvent(data_slice -> streams[peer_ + num_gpus],
                    data_slice -> middle_events[0], 0),
                    "cudaStreamWaitEvent failed", __FILE__, __LINE__))
                    break;
                PushNeighbor<Enactor, GraphSliceT, DataSlice, 1, 1>(
                    enactor,
                    thread_num,
                    peer,
                    data_slice -> local_vertices.GetSize(),
                    enactor_stats,
                    problem -> data_slices[thread_num].GetPointer(util::HOST),
                    problem -> data_slices[peer      ].GetPointer(util::HOST),
                    problem -> graph_slices[thread_num],
                    problem -> graph_slices[peer      ],
                    data_slice -> streams[peer_ + num_gpus],
                    enactor -> communicate_multipy);
                //Set_Record(data_slice, enactor_stats->iteration, peer_, 0, data_slice->strams[peer_]);
                if (enactor_stats -> retval = util::GRError(cudaEventRecord(
                    data_slice -> middle_events[peer_], data_slice -> streams[peer_ + num_gpus]),
                    "cudaEventRecord failed", __FILE__, __LINE__))
                    break;
                data_slice -> middle_event_set[peer_] = true;
            }
            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Pushed",
                    thread_num, enactor_stats -> iteration); 

            
            //printf("%d events set\n", thread_num);fflush(stdout);
            has_error =false;
            for (int i=0; i<num_gpus * num_gpus; i++)
            {
                if (s_enactor_stats[i].retval != cudaSuccess)
                {
                    has_error = true;
                    break;
                }
            }
            if (has_error)
            {
                thread_data -> status = ThreadSlice::Status::Idle;
                continue;
            }

            for (int peer_ = 1; peer_ < num_gpus; peer_ ++)
            {
                data_slice -> keys_out            [peer_].ForceSetPointer(
                    data_slice -> keys_outs            [peer_], util::DEVICE);
                data_slice -> vertex_associate_out[peer_].ForceSetPointer(
                    data_slice -> vertex_associate_outs[peer_], util::DEVICE);
                data_slice -> value__associate_out[peer_].ForceSetPointer(
                    data_slice -> value__associate_outs[peer_], util::DEVICE);
            }

            for (int peer = 0; peer < num_gpus; peer++)
                middle_event_markers[peer] = 0;
            middle_event_markers[thread_num] = 1;
            middle_event_counter = 1;
            while (middle_event_counter < num_gpus)
            {
                for (int peer = 0; peer < num_gpus; peer++)
                {
                    if (middle_event_markers[peer] == 1) continue;
                    int peer_ = peer < thread_num ? peer + 1 : peer;
                    int gpu_  = peer < thread_num ? thread_num : thread_num + 1;
                    if (!s_data_slice[peer] -> middle_event_set[gpu_]) continue;
                    
                    s_data_slice[peer] -> middle_event_set[gpu_] = false;
                    if (enactor_stats -> retval = util::GRError(
                        cudaStreamWaitEvent(data_slice -> streams[peer_],
                            s_data_slice[peer] -> middle_events[gpu_], 0),
                        "cudaStramWaitEvent failed", __FILE__, __LINE__))
                        break;
                    SizeT num_elements = s_data_slice[peer] -> local_vertices.GetSize();
                    num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2 + 1;
                    VertexId peer_iteration = s_data_slice[peer] -> middle_iteration;
                    if (num_blocks > 480) num_blocks = 480;
                    
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys_in", data_slice -> keys_in[peer_iteration%2][peer_].GetPointer(util::DEVICE), num_elements, thread_num, -1, peer_, data_slice -> streams[peer_]);
                    Expand_Incoming_Middle<VertexId, SizeT, Value>
                        <<<num_blocks, AdvanceKernelPolicy::THREADS, 
                        0, data_slice -> streams[peer_]>>>(
                        num_elements,
                        data_slice -> keys_in            [peer_iteration % 2][peer_].GetPointer(util::DEVICE),
                        data_slice -> vertex_associate_in[peer_iteration % 2][peer_].GetPointer(util::DEVICE),
                        data_slice -> value__associate_in[peer_iteration % 2][peer_].GetPointer(util::DEVICE),
                        data_slice -> labels.GetPointer(util::DEVICE),
                        data_slice -> sigmas.GetPointer(util::DEVICE));

                    middle_event_markers[peer] = 1;
                    middle_event_counter ++;
                }
                
                has_error =false;
                for (int i=0; i<num_gpus * num_gpus; i++)
                {
                    if (s_enactor_stats[i].retval != cudaSuccess)
                    {
                        has_error = true;
                        break;
                    }
                }
                if (has_error)
                {
                    thread_data -> status = ThreadSlice::Status::Idle;
                    break;
                }

                if (middle_event_counter < num_gpus)
                    sleep(0);
            }
            if (has_error) continue;
            //printf("%d events clear\n", thread_num);fflush(stdout);

            //data_slice->sigmas.Move(util::DEVICE, util::HOST);
            //data_slice->labels.Move(util::DEVICE, util::HOST);

            //CPU_Barrier;
            //util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[0],thread_num);
            //barrier_markers[0][thread_num] = 1;
            //bool barrier_pass = false;
            //while (!barrier_pass)
            //{
            //    barrier_pass = true;
            //    for (int peer_ = 0; peer_ < num_gpus; peer_ ++)
            //    if (barrier_markers[0][peer_] == 0)
            //    {
            //        barrier_pass = false;
            //        break;
            //    }
            //    if (!barrier_pass) sleep(0);
            //}

            long max_iteration=0;
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                if (s_data_slice[gpu] -> middle_iteration > max_iteration)
                    max_iteration = s_data_slice[gpu] -> middle_iteration;
            }
            while (data_slice -> forward_queue_offsets[0].size() < max_iteration)
            {
                data_slice -> forward_queue_offsets[0].push_back(
                    data_slice -> forward_queue_offsets[0].back());
            }
            enactor_stats[0].iteration=max_iteration;
            //for (VertexId node = 0; node < graph_slice->in_counter[0]; node++)
            //{
            //    for (SizeT i = graph_slice -> backward_offset[node];
            //        i < graph_slice -> backward_offset[node+1]; i++)
            //    {
            //        int peer = graph_slice -> backward_partition[i];
            //        if (peer <= thread_num) peer--;
            //        int _node = graph_slice -> backward_convertion[i];
            //        s_data_slice[peer] -> sigmas[_node] = data_slice -> sigmas[node];
            //        s_data_slice[peer] -> labels[_node] = data_slice -> labels[node];
            //    }
            //}
            for (int peer_ = 1; peer_ < num_gpus; peer_ ++)
            {
                if (enactor_stats -> retval = util::GRError(
                    cudaStreamSynchronize(data_slice -> streams[peer_]),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                    break;
            }

            // CPU barrier
            data_slice -> middle_finish = true;
            has_error = false;
            for (int peer = 0; peer < num_gpus; peer++)
            {
                while (!s_data_slice[peer] -> middle_finish && !has_error)
                {
                    sleep(0);
                    for (int i=0; i<num_gpus * num_gpus; i++)
                    if (s_enactor_stats[i].retval != cudaSuccess)
                    {
                        has_error = true; break;
                    }
                }
                if (has_error) break;
            }
            if (has_error) continue;

            for (int gpu = 0; gpu < num_gpus * 2; gpu++)
                data_slice -> wait_marker[gpu] = 0; 
            for (int i=0; i<4; i++) 
            for (int gpu = 0; gpu < num_gpus * 2; gpu++)
            for (int stage=0; stage < data_slice -> num_stages; stage++)
                data_slice -> events_set[i][gpu][stage] = false;
            for (int gpu = 0; gpu < num_gpus; gpu++)
            for (int i=0; i<2; i++) 
                data_slice -> in_length[i][gpu] = 0; 
            for (int peer = 0; peer < num_gpus; peer++)
            {
                data_slice -> out_length[peer] = 1; 
                data_slice -> first_backward_incoming[peer] = true;
            }

            //CPU_Barrier;
            //util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[1], thread_num);
            //barrier_markers[1][thread_num] = 1;
            //barrier_pass = false;
            //while (!barrier_pass)
            //{
            //    barrier_pass = true;
            //    for (int peer_ = 0; peer_ < num_gpus; peer_ ++)
            //    if (barrier_markers[1][peer_] == 0)
            //    {
            //        barrier_pass = false;
            //        break;
            //    }
            //    if (!barrier_pass) sleep(0);
            //}

            //data_slice -> sigmas.Move(util::HOST, util::DEVICE);
            //data_slice -> labels.Move(util::HOST, util::DEVICE);
            data_slice -> value__associate_orgs[0] = 
                data_slice -> deltas.GetPointer(util::DEVICE);
            data_slice -> value__associate_orgs[1] = 
                data_slice -> bc_values.GetPointer(util::DEVICE);
            data_slice -> value__associate_orgs.Move(util::HOST, util::DEVICE);
        } else {
        }
        
        has_error =false;
        for (int i=0; i<num_gpus * num_gpus; i++)
        {
            if (s_enactor_stats[i].retval != cudaSuccess)
            {
                has_error = true;
                break;
            }
        }
        if (has_error)
        {
            thread_data -> status = ThreadSlice::Status::Idle;
            continue;
        }

        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Backward begin", 
                thread_num, enactor_stats -> iteration);
        gunrock::app::Iteration_Loop
            <Enactor, BcBFunctor, 
            Backward_Iteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>, 
            0, 2> (thread_data);
        if (enactor -> debug) 
            util::cpu_mt::PrintMessage("Backward phase finished.", 
                thread_num, enactor_stats->iteration);
       thread_data -> status = ThreadSlice::Status::Idle;
    }

    thread_data->status = ThreadSlice::Status::Ended;

    CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 *
 * @tparam _Problem Problem type we process on
 * @tparam _INSTRUMENT Whether or not to collect per-CTA clock-count stats.
 * @tparam _DEBUG Whether or not to enable debug mode.
 * @tparam _SIZE_CHECK Whether or not to enable size check.
 */
template<typename _Problem /*, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK*/>
class BCEnactor :
    public EnactorBase<typename _Problem::SizeT/*, _DEBUG, _SIZE_CHECK*/>
{
     // Members
    ThreadSlice *thread_slices;
    CUTThread   *thread_Ids   ;
    util::cpu_mt::CPUBarrier *cpu_barrier;

    // Methods
public:
    _Problem    *problem      ;
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    typedef BCEnactor<Problem>         Enactor;
    //static const bool INSTRUMENT = _INSTRUMENT;
    //static const bool DEBUG      = _DEBUG;
    //static const bool SIZE_CHECK = _SIZE_CHECK;
    util::Array1D<int, unsigned char> barrier_markers[2];

   /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BCEnactor constructor
     */
    BCEnactor(
        int   num_gpus   = 1, 
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(
            VERTEX_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check)
    {
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
        cpu_barrier   = NULL;
        barrier_markers[0].SetName("barrier_markers[0]");
        barrier_markers[1].SetName("barrier_markers[1]");
    }

    /**
     *  @brief BCenactor destructor
     */
    virtual ~BCEnactor()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (thread_slices != NULL)
        {
            for (int gpu = 0; gpu < this -> num_gpus; gpu++)
                thread_slices[gpu].status = ThreadSlice::Status::ToKill;

            cutWaitForThreads(thread_Ids, this->num_gpus);
            delete[] thread_Ids   ; thread_Ids    = NULL;
            delete[] thread_slices; thread_slices = NULL;
        }
        if (retval = BaseEnactor::Release()) return retval;
        if (retval = barrier_markers[0].Release()) return retval;
        if (retval = barrier_markers[1].Release()) return retval;
        problem = NULL;
        //if (cpu_barrier!=NULL)
        //{
        //    util::cpu_mt::DestoryBarrier(&cpu_barrier[0]);
        //    util::cpu_mt::DestoryBarrier(&cpu_barrier[1]);
        //    delete[] cpu_barrier;cpu_barrier=NULL;
        //}
        return retval;
    }

    /**
     * @brief Initialize the problem.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolity,
        typename FilterKernelPolicy>
    cudaError_t InitBC(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 512)
    {
        cudaError_t retval = cudaSuccess;
        // Lazy initialization
        if (retval = BaseEnactor::Init(
            max_grid_size,
            AdvanceKernelPolity::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) 
            return retval;

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];
        barrier_markers[0].Allocate(this -> num_gpus);
        barrier_markers[1].Allocate(this -> num_gpus);

        /*for (int gpu=0;gpu<this->num_gpus;gpu++)
        {    
            if (retval = util::SetDevice(this->gpu_idx[gpu])) break;
            if (sizeof(SizeT) == 4)
            {
                cudaChannelFormatDesc row_offsets_dest = cudaCreateChannelDesc<SizeT>();
                gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets.channelDesc = row_offsets_dest;
                if (retval = util::GRError(cudaBindTexture( 
                    0,
                    gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,
                    problem->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE),
                    ((size_t) (problem -> graph_slices[gpu]->nodes + 1)) * sizeof(SizeT)),
                    "BFSEnactor cudaBindTexture row_offsets_ref failed",
                    __FILE__, __LINE__)) break;
            }
        }            
        if (retval) return retval;*/

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            //thread_slices[gpu].cpu_barrier  = cpu_barrier;
            thread_slices[gpu].thread_num   = gpu;
            thread_slices[gpu].problem      = (void*)problem;
            thread_slices[gpu].enactor      = (void*)this;
            thread_slices[gpu].context      =&(context[gpu*this->num_gpus]);
            problem -> data_slices[gpu] -> barrier_markers = barrier_markers;
            thread_slices[gpu].status       = ThreadSlice::Status::Inited;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(BCThread<
                    AdvanceKernelPolity, FilterKernelPolicy,
                    BCEnactor<Problem> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }

        for (int gpu=0; gpu < this->num_gpus; gpu++)
        {    
            while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
            {    
                sleep(0);
                //std::this_thread::yield();
            }    
        } 
        return retval;
    }

    /**
     * @brief Reset enactor
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseEnactor::Reset())
            return retval;
        for (int gpu=0; gpu < this->num_gpus; gpu++)
        {
            thread_slices[gpu].status = ThreadSlice::Status::Wait;
            barrier_markers[0][gpu] = 0;
            barrier_markers[1][gpu] = 0;
        }
        return retval;
    }

    /** @} */

    /**
     * @brief Enacts a betweenness-centrality computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] src Source node to start primitive.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactBC(VertexId    src)
    {
        cudaError_t              retval         = cudaSuccess;

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
            {
                thread_slices[gpu].init_size=1;
            } else thread_slices[gpu].init_size=0;
            //this->frontier_attribute[gpu*this->num_gpus].queue_length = thread_slices[gpu].init_size;
        }

        for (int gpu=0; gpu< this->num_gpus; gpu++)
        {    
            thread_slices[gpu].status = ThreadSlice::Status::Running;
        }    
        for (int gpu=0; gpu< this->num_gpus; gpu++)
        {    
            while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
            {    
                sleep(0);
                //std::this_thread::yield();
            }    
        }    

        for (int gpu=0; gpu<this->num_gpus * this -> num_gpus;gpu++)
        if (this->enactor_stats[gpu].retval!=cudaSuccess)
        {    
            retval=this->enactor_stats[gpu].retval;
            return retval;
        }    

        if (this -> debug) printf("\nGPU BC Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
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
        2,                                  // MIN_CTA_OCCUPANCY
        9,                                 // LOG_THREADS
        10,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LB_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        2,                                  // MIN_CTA_OCCUPANCY
        9,                                 // LOG_THREADS
        10,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    LB_LIGHT_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        2,                                  // MIN_CTA_OCCUPANCY
        9,                                 // LOG_THREADS
        10,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_CULL>
    LB_CULL_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        2,                                  // MIN_CTA_OCCUPANCY
        9,                                 // LOG_THREADS
        10,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT_CULL>
    LB_LIGHT_CULL_AdvanceKernelPolicy;

    template <typename Dummy, gunrock::oprtr::advance::MODE A_MODE>
    struct MODE_SWITCH{};

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBC<LB_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitBC <LB_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    /*template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::TWC_FORWARD>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBC<TWC_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitBC <TWC_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };*/

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBC<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitBC <LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_CULL>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBC<LB_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitBC <LB_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT_CULL>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBC<LB_LIGHT_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitBC <LB_LIGHT_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    /**
     * @brief BC Enact kernel entry.
     *
     * @param[in] src Source node to start primitive.
     * @param[in] traversal_mode Mode of workload strategy in advance
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Enact(
        VertexId src,
        std::string traversal_mode = "LB")
    {
        if (this -> min_sm_version >= 300)
        {
            //return EnactBC<AdvanceKernelPolicy, FilterKernelPolicy>(src);
            if (traversal_mode == "LB")
                return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>
                    ::Enact(*this, src);
            //else if (traversal_mode == "TWC")
            //     return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>
            //        ::Enact(*this, src);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_CULL>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Enact(*this, src);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief BC Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] traversal_mode Mode of workload strategy in advance
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr *context,
        Problem    *problem,
        int max_grid_size = 512,
        std::string traversal_mode = "LB")
    {
        if (this -> min_sm_version >= 300)
        {
            //return InitBC<AdvanceKernelPolicy, FilterKernelPolicy>(
            //    context, problem, max_grid_size);
            if (traversal_mode == "LB")
                return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>
                    ::Init(*this, context, problem, max_grid_size);
            //else if (traversal_mode == "TWC")
            //     return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>
            //        ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else printf("Traversal mode %s is not supported by BC at the moment\n", traversal_mode.c_str());
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
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
