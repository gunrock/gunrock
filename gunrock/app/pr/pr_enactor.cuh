// ---------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ---------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------------------

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
#include <gunrock/util/sharedmem.cuh>
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/app/pr/pr_functor.cuh>
#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace pr {

/*
 * @brief Make_Queue function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements
 * @param[in] keys_in
 * @param[in] marker
 * @param[out] keys_out
 */
template <
    typename VertexId,
    typename SizeT>
__global__ void Mark_Queue_R0D (
    const SizeT           num_elements,
    const VertexId* const keys_in,
    const SizeT*    const degrees,
          SizeT*          marker)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    VertexId x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

    while ( x < num_elements)
    {
        VertexId key = keys_in[x];
        //if (degrees[key] == 0) printf("d[%d @ %d]==0 \t", key, x);
        marker[x] = degrees[key]==0? 1 :0;
        x += STRIDE;
    }
}

/*
 * @brief Make_Queue function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements
 * @param[in] keys_in
 * @param[in] marker
 * @param[out] keys_out
 */
template <
    typename VertexId,
    typename SizeT>
__global__ void Make_Queue_R0D (
    const SizeT           num_elements,
    const VertexId* const keys_in,
    const SizeT*    const marker,
          VertexId*       keys_out)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    VertexId x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

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
 * @param[in] degrees
 */
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
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    VertexId x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
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
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    VertexId x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
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
    typename Value>
__global__ void Selective_Reset_PR(
    const SizeT        num_elements,
    const VertexId* const keys,
    const Value        reset_value,
    const SizeT* const markers,
    const Value* const rank_next,
          Value*       rank_curr)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys[x];
        if (markers[key] == 0)
            rank_curr[key] = reset_value;
        else
            rank_curr[key] = rank_next[key];
        x += STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT,
    typename Value>
__global__ void Selective_Reset_PR2(
    const SizeT        num_elements,
    const VertexId* const keys,
    const SizeT* const markers,
    const Value* const rank_next,
          Value*       rank_curr)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys[x];
        if (markers[key] == 1)
            rank_curr[key] = rank_next[key];
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
 * @param[in] array_size
 * @param[in] array
 */
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
          char*           array,
          SizeT*          markers)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
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

    x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys_in[x];
        Value add_value = s_value__associate_in[0][x];
        //if (isfinite(add_value))
        //{
        Value old_value = atomicAdd(s_value__associate_org[0] + key, add_value);
        markers[key] = 1;
            //if (to_track(key)) printf("rank[%d] = %.8le + %.8le = %.8le\n", 
            //    key, old_value, s_value__associate_in[0][x], old_value + s_value__associate_in[0][x]);
        //}
        x+=STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT>
__global__ void Assign_Marker_PR(
    const SizeT     num_elements,
    const int       peer_,
    const SizeT*    markers,
    const int*      partition_table,
          SizeT*    key_markers)
{
    //extern __shared__ SizeT* s_marker[];
    //SharedMemory<SizeT*> smem;
    //SizeT** s_marker = smem.getPointer();
    int   gpu = 0;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    //if (threadIdx.x < num_gpus)
    //    s_marker[threadIdx.x] = key_markers[threadIdx.x];
    //__syncthreads();

    while (x < num_elements)
    {
        //gpu = num_gpus;
        gpu = partition_table[x];
        //if (markers[x] != 1 && gpu != 0)
        //{
        //    gpu = num_gpus;
        //}
        //for (int i=0; i<num_gpus; i++)
        //    s_marker[i][x] = (i==gpu)?1:0;
        if ((markers[x] == 1 || gpu == 0) && (gpu == peer_))
            key_markers[x] = 1;
        else key_markers[x] = 0;
        x+=STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT>
__global__ void Assign_Keys_PR (
    const SizeT          num_elements,
    const int            peer_,
    const int*           partition_table,
    const SizeT*         markers,
          SizeT*         keys_marker,
          VertexId*      keys_out)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

    while (x < num_elements)
    {
        int gpu = partition_table[x];
        if ((markers[x] == 1 || gpu == 0) && (gpu == peer_))
        {
            //if (gpu > 0)
            //{
                SizeT pos = keys_marker[x]-1;
                //printf("keys_outs[%d][%d] <- %d \t", gpu, pos, x);
                keys_out[pos] = x;
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
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

    while (x < num_elements)
    {
        VertexId key = keys_out[x];
        rank_out[x] = rank_next[key];
        x+=STRIDE;
    }
}

/*
 * @brief Expand incoming function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] num_elements
 * @param[in] keys_in
 * @param[in] ranks_in
 * @param[in] ranks_out
 */
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
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys_in[x];
        ranks_out[key] = ranks_in[x];
        x+=STRIDE;
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
    typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value> 
                                         Frontier  ;
    typedef IterationBase <
        AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        false, true, false, true, false> BaseIteration;
    typedef RemoveZeroDegreeNodeFunctor<
            VertexId,
            SizeT,
            Value,
            Problem> RemoveZeroFunctor;

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
    util::DoubleBuffer<SizeT, VertexId, Value>
                                  *frontier_queue,
    util::Array1D<SizeT, SizeT>   *scanned_edges,
    FrontierAttribute<SizeT>      *frontier_attribute,
    EnactorStats                  *enactor_stats,
    DataSlice                     *data_slice,
    DataSlice                     *d_data_slice,
    GraphSliceT                   *graph_slice,
    util::CtaWorkProgressLifetime *work_progress,
    ContextPtr                     context,
    cudaStream_t                   stream)
{
    if (enactor_stats->iteration == 0)
    {
        frontier_attribute->queue_reset  = true;
        frontier_attribute->selector     = 0;
        frontier_attribute->queue_index  = 0;
        frontier_attribute->queue_length = 
            data_slice->num_gpus > 1 ? 
            data_slice->local_nodes : graph_slice->nodes;
    }
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
    EnactorStats                  *enactor_stats,
    DataSlice                     *data_slice,
    DataSlice                     *d_data_slice,
    GraphSliceT                   *graph_slice,
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

    //bool over_sized = false;
    //if (enactor_stats->retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT>(
    //    "scanned_edges", frontier_attribute->queue_length, scanned_edges, over_sized, thread_num, enactor_stats->iteration, peer_)) return;
    //if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, stream)) return;
    frontier_attribute->queue_reset = true;
    enactor_stats -> nodes_queued[0] += frontier_attribute -> queue_length;
    gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, RemoveZeroFunctor>(
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
        false,
        false);

    //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
    //      "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
    enactor_stats      -> AccumulateEdges(
        work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(frontier_attribute->queue_index+1), stream);

    gunrock::oprtr::filter::LaunchKernel
        <FilterKernelPolicy, Problem, RemoveZeroFunctor>(
        enactor_stats->filter_grid_size, 
        FilterKernelPolicy::THREADS, 
        (size_t)0, 
        stream,
        enactor_stats->iteration,
        frontier_attribute->queue_reset,
        frontier_attribute->queue_index,
        frontier_attribute->queue_length,
        frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
        (Value*)NULL,
        frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
        d_data_slice,
        (unsigned char*)NULL,
        work_progress[0],
        frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
        frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
        enactor_stats->filter_kernel_stats);

    //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
    //      "filter::Kernel RemoveZeroFunctor failed", __FILE__, __LINE__)))
    //    break;

    Clear_Zero_R0D <VertexId, SizeT>
        <<<128, 128, 0, stream>>> (
        graph_slice->nodes,
        data_slice -> degrees.GetPointer(util::DEVICE),
        data_slice -> degrees_pong.GetPointer(util::DEVICE));

    util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
        data_slice->degrees.GetPointer(util::DEVICE),
        data_slice->degrees_pong.GetPointer(util::DEVICE), graph_slice->nodes);

    //util::DisplayDeviceResults(problem->data_slices[0]->d_degrees,
    //        graph_slice->nodes);

    frontier_attribute -> queue_index++;
    frontier_attribute -> selector^=1;
    if (enactor_stats  -> retval = 
        work_progress  -> GetQueueLength(
            frontier_attribute -> queue_index, 
            frontier_attribute -> queue_length, 
            false, stream)) return;
    if (enactor_stats->retval = util::GRError(
        cudaStreamSynchronize(stream), 
       "cudaStreamSynchronize failed", __FILE__, __LINE__)) return;
    //enactor_stats->total_queued[0] += frontier_attribute->queue_length;
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys1", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
    //util::cpu_mt::PrintGPUArray<SizeT, SizeT>("degrees1", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);

    if (num_valid_node == frontier_attribute->queue_length || num_valid_node==0) 
         data_slice->to_continue = false;
    else data_slice->to_continue = true;
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
    if (retval = Check_Size<SizeT, SizeT> (
        enactor -> size_check, "scanned_edges", 
        frontier_attribute->queue_length, 
        partitioned_scanned_edges, over_sized, 
        -1, -1, -1, false)) return retval;
    retval = gunrock::oprtr::advance::ComputeOutputLength
        <AdvanceKernelPolicy, Problem, RemoveZeroFunctor>(
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
        ADVANCE_TYPE,
        express,
        in_inv,
        out_inv);
    return retval;
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
static void Expand_Incoming(
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
    Expand_Incoming_R0D
        <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
        <<<grid_size, block_size, shared_size, stream>>> (
        num_elements,
        keys_in,
        data_slice->degrees.GetPointer(util::DEVICE));
    num_elements = 0;
}

static bool Stop_Condition(
    EnactorStats                    *enactor_stats,
    FrontierAttribute<SizeT>        *frontier_attribute,
    util::Array1D<SizeT, DataSlice> *data_slice,
    int num_gpus)
{
    //printf("CC Stop checked\n");fflush(stdout);
    for (int gpu = 0; gpu < num_gpus*num_gpus; gpu++)
    if (enactor_stats[gpu].retval != cudaSuccess)
    {
        printf("(CUDA error %d @ GPU %d: %s\n", 
            enactor_stats[gpu].retval, gpu % num_gpus, 
            cudaGetErrorString(enactor_stats[gpu].retval)); 
        fflush(stdout);
        return true;
    }

    /*for (int gpu = 0; gpu< num_gpus*num_gpus; gpu++)
    if (enactor_stats[gpu].iteration == 0)
    {
        printf("enactor_stats[%d].iteration ==0\n", gpu);fflush(stdout);
        return false;
    }*/

    bool past_max_iter = true;
    for (int gpu=0; gpu<num_gpus*num_gpus; gpu++)
    if (enactor_stats[gpu].iteration < data_slice[0].max_iter)
    {
        past_max_iter = false;
        break;
    }
    if (past_max_iter) return true;

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

/*
 * @brief Make_Output function.
 *
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] thread_num Number of threads.
 * @param[in] num_elements
 * @param[in] num_gpus Number of GPUs used.
 * @param[in] frontier_queue Pointer to the frontier queue.
 * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
 * @param[in] frontier_attribute Pointer to the frontier attribute.
 * @param[in] enactor_stats Pointer to the enactor statistics.
 * @param[in] data_slice Pointer to the data slice we process on.
 * @param[in] graph_slice Pointer to the graph slice we process on.
 * @param[in] work_progress Pointer to the work progress class.
 * @param[in] context CudaContext for ModernGPU API.
 * @param[in] stream CUDA stream.
 */
template <
    int NUM_VERTEX_ASSOCIATES,
    int NUM_VALUE__ASSOCIATES>
static void Make_Output(
    Enactor                       *enactor,
    int                            thread_num,
    SizeT                          num_elements,
    int                            num_gpus,
    Frontier                      *frontier_queue,
    util::Array1D<SizeT, SizeT>   *scanned_edges,
    FrontierAttribute<SizeT>      *frontier_attribute,
    EnactorStats                  *enactor_stats,
    util::Array1D<SizeT, DataSlice>
                                  *data_slice,
    GraphSliceT                   *graph_slice,
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
        (SizeT*)data_slice[0] -> markers.GetPointer(util::DEVICE),
        num_elements,
        (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
        (SizeT*)data_slice[0] -> markers.GetPointer(util::DEVICE),
        context[0]);

    Make_Queue_R0D <VertexId, SizeT>
        <<<grid_size, block_size, 0, stream>>> (
        num_elements,
        frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
        data_slice[0]->markers.GetPointer(util::DEVICE),
        data_slice[0]->keys_out[1].GetPointer(util::DEVICE));

    if (!enactor -> size_check)
        util::MemsetCopyVectorKernel <<<grid_size, block_size, 0, stream>>>(
            data_slice[0]->frontier_queues[0].keys[frontier_attribute->selector].GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
            num_elements);

    cudaMemcpyAsync(&data_slice[0]->out_length[1], 
        data_slice[0]->markers.GetPointer(util::DEVICE) + num_elements -1, 
        sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
    //printf("num_lements = %d data_slice[%d]->out_length[1] = %d\n", num_elements, thread_num, data_slice[0]->out_length[1]);fflush(stdout);
    if (enactor_stats->retval = util::GRError(
        cudaStreamSynchronize(stream), 
       "cudaStramSynchronize failed", __FILE__, __LINE__)) return;
    //printf("num_lements = %d data_slice[%d]->out_length[1] = %d\n", num_elements, thread_num, data_slice[0]->out_length[1]);fflush(stdout);
    for (peer_ = 2; peer_ < num_gpus; peer_++)
        data_slice[0]->out_length[peer_] = data_slice[0]->out_length[1];
    data_slice[0]->out_length[0] = frontier_attribute->queue_length;
}

/*static void Check_Queue_Size(
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
    }
} */

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
    typedef GraphSlice     <VertexId, SizeT, Value> GraphSliceT;
    typedef PRFunctor      <VertexId, SizeT, Value, Problem> PrFunctor;
    typedef PRMarkerFunctor<VertexId, SizeT, Value, Problem> PrMarkerFunctor;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value>
                                         Frontier  ;
    typedef IterationBase <
        AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        false, true, false, true, false> BaseIteration;

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
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        //Print_Const<DataSlice><<<1,1,0,stream>>>(d_data_slice);
        //for (int i=0; i<3; i++)
        //{
        //if (enactor_stats -> iteration != 0 || i!=0)
        if (enactor_stats -> iteration != 0)
        {
            frontier_attribute->queue_length = data_slice -> edge_map_queue_len;
            enactor_stats->edges_queued[0] += frontier_attribute->queue_length;

            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Filter start.",
                    thread_num, enactor_stats->iteration, peer_);
             // filter kernel
            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, Problem, PrFunctor>(
                enactor_stats->filter_grid_size, 
                FilterKernelPolicy::THREADS, 
                (size_t)0, 
                stream,
                enactor_stats->iteration,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                (Value*)NULL,
                (VertexId*)NULL,//frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),// d_out_queue
                d_data_slice,
                (unsigned char*)NULL,
                work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),           // max_in_queue
                util::MaxValue<VertexId>(), //frontier_queue->keys[frontier_attribute->selector^1].GetSize(),         // max_out_queue
                enactor_stats->filter_kernel_stats);

            //if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
            //cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

            //printf("Filter end.\n");fflush(stdout);
            //enactor_stats->iteration++;
            frontier_attribute->queue_index++;

            if (enactor_stats->retval = work_progress -> GetQueueLength(
                frontier_attribute->queue_index, 
                frontier_attribute->queue_length, 
                false, stream)) 
                return;
            //num_elements = queue_length;

            //swap rank_curr and rank_next
            //util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            //    data_slice->rank_curr.GetPointer(util::DEVICE),
            //    data_slice->rank_next.GetPointer(util::DEVICE),
            //    graph_slice->nodes);
            /*if (enactor_stats -> iteration == 1)
                Selective_Reset_PR <<<256, 256, 0, stream>>>(
                    data_slice -> edge_map_queue_len,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                    data_slice -> reset_value,
                    data_slice -> markers  .GetPointer(util::DEVICE),
                    data_slice -> rank_next.GetPointer(util::DEVICE),
                    data_slice -> rank_curr.GetPointer(util::DEVICE));
            else
                Selective_Reset_PR2<<<256, 256, 0, stream>>>(
                    data_slice -> edge_map_queue_len,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                    data_slice -> markers  .GetPointer(util::DEVICE),
                    data_slice -> rank_next.GetPointer(util::DEVICE),
                    data_slice -> rank_curr.GetPointer(util::DEVICE));*/
     
            util::MemsetKernel<<<256, 256, 0, stream>>>(
                data_slice->rank_next.GetPointer(util::DEVICE),
                (Value)0.0, graph_slice->nodes);

            if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), 
                "cudaStreamSynchronize failed", __FILE__, __LINE__)) 
                return;
            data_slice->PR_queue_length = frontier_attribute->queue_length;

            //enactor_stats      -> Accumulate(
            //    work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(frontier_attribute->queue_index), stream);
            //printf("queue_length = %d\n", frontier_attribute->queue_length);fflush(stdout);
            //if (false) {
            //    if (INSTRUMENT || DEBUG) {
                //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
                //enactor_stats->total_queued += frontier_attribute->queue_length;
                //if (DEBUG) printf(", %lld", (long long) frontier_attribute.queue_length);
            //    if (Enactor::INSTRUMENT) {
            //        if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
            //            enactor_stats->filter_grid_size,
            //            enactor_stats->total_runtimes,
            //            enactor_stats->total_lifetimes,
            //            false, stream)) return;
            //    }
            //}
        }

        //if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index, edge_map_queue_len)) return;
        frontier_attribute->queue_length = data_slice->edge_map_queue_len;
        //if (enactor_stats->iteration == 0) util::cpu_mt::PrintGPUArray("keys", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, peer_, stream);
        //if (enactor_stats->iteration == 0) util::cpu_mt::PrintGPUArray<SizeT, SizeT>("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);
        //util::cpu_mt::PrintGPUArray<SizeT, Value>("ranks", data_slice->rank_curr.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, peer_, stream);

        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Advance Prfunctor start.",
                thread_num, enactor_stats -> iteration, peer_);

        // Edge Map
        frontier_attribute->queue_reset = true;
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
            (VertexId*)NULL, //frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
            (Value*   )NULL,
            (Value*   )NULL,
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
            false,
            false);

        //if (enactor_stats->retval = work_progress->GetQueueLength(
        //    frontier_attribute->queue_index+1, 
        //    frontier_attribute->queue_length, 
        //    false, stream, true)) 
        //    return;
        //enactor_stats->edges_queued[0] += frontier_attribute->queue_length;
        enactor_stats -> AccumulateEdges(
            work_progress -> GetQueueLengthPointer<unsigned int, SizeT>(
                frontier_attribute -> queue_index + 1), stream);
        frontier_attribute -> queue_length = data_slice->edge_map_queue_len;

        if (enactor_stats -> iteration == 0)
        {
            util::MemsetKernel<<<256, 256, 0, stream>>>(
                data_slice->markers.GetPointer(util::DEVICE), 
                (SizeT)0, graph_slice->nodes);
            //util::cpu_mt::PrintGPUArray("keys", frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration, -1, stream);
            //util::cpu_mt::PrintGPUArray("row_offsets", graph_slice->row_offsets.GetPointer(util::DEVICE), graph_slice->nodes+1, thread_num, enactor_stats->iteration, -1, stream);
            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Advance PrMarkerFunctor start.",
                    thread_num, enactor_stats -> iteration, peer_);
            frontier_attribute -> queue_reset = true;
            // Edge Map
            gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, PrMarkerFunctor>(
                //d_done,
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges->GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), // d_in_queue
                (VertexId*)NULL, //frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
                (Value*   )NULL,
                (Value*   )NULL,
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
                false,
                true);
            //printf("Advance end.\n");fflush(stdout);
            //util::cpu_mt::PrintGPUArray("markers", data_slice[0]->markers.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, -1, stream);
        }
        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize failed", __FILE__, __LINE__)) 
            return;
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
        //}
    }

    static cudaError_t Compute_OutputLength(
        Enactor                    *enactor,
        FrontierAttribute<SizeT>   *frontier_attribute,
        SizeT                      *d_offsets,
        VertexId                   *d_indices,
        SizeT                      *d_inv_offsets,
        VertexId                   *d_inv_indices,
        VertexId                   *d_in_key_queue,
        util::Array1D<SizeT,SizeT> *partitioned_scanned_edges,
        SizeT                       max_in,
        SizeT                       max_out,
        CudaContext                &context,
        cudaStream_t                stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                        express = false,
        bool                        in_inv  = false,
        bool                        out_inv = false)
    {
        //printf("Compute_OutputLength start.\n");fflush(stdout);
        cudaError_t retval = cudaSuccess;
        if (AdvanceKernelPolicy::ADVANCE_MODE ==  gunrock::oprtr::advance::TWC_FORWARD)
        {
            //return retval;
        } else {
            bool over_sized = false;
            if (retval = Check_Size<SizeT, SizeT> (
                enactor -> size_check, "scanned_edges", 
                frontier_attribute->queue_length, 
                partitioned_scanned_edges, 
                over_sized, -1, -1, -1, false)) return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, PrFunctor>(
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
                ADVANCE_TYPE,
                express,
                in_inv,
                out_inv);
        }
        //printf("Compute_OutputLength end.\n");fflush(stdout);
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
        EnactorStats                  *enactor_stats,
        GraphSliceT                   *graph_slice)
    { 
        return ; // no need to check queue size for PR
    }

    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
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
        //util::cpu_mt::PrintCPUArray("Incoming_length", &num_elements, 1, data_slice->gpu_idx);
        Expand_Incoming_PR
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            array_size,
            array,
            data_slice -> markers.GetPointer(util::DEVICE));
        num_elements = 0;
    }

    /*
     * @brief Stop_Condition check function.
     *
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] num_gpus Number of GPUs used.
     */
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
        Enactor                       *enactor,
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice,
        GraphSliceT                   *graph_slice,
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
            SizeT temp_length = data_slice[0]->out_length[0];
            for (peer_ = 0; peer_<num_gpus; peer_++)
            {
                util::MemsetKernel<<<128, 128, 0, stream>>> ( 
                    data_slice[0]->keys_marker[0].GetPointer(util::DEVICE), 
                    (SizeT)0, graph_slice->nodes);

                Assign_Marker_PR<VertexId, SizeT>
                    <<<grid_size, block_size, 0, stream>>> (
                    graph_slice->nodes,
                    peer_,
                    data_slice[0]->markers.GetPointer(util::DEVICE),
                    graph_slice->partition_table.GetPointer(util::DEVICE),
                    data_slice[0]->keys_marker[0].GetPointer(util::DEVICE));
                //for (peer_ = 0; peer_<num_gpus;peer_++)
                //    util::cpu_mt::PrintGPUArray("keys_marker0", data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, -1, stream);

                Scan<mgpu::MgpuScanTypeInc>(
                    (SizeT*)(data_slice[0]->keys_marker[0].GetPointer(util::DEVICE)),
                    graph_slice->nodes,
                    (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
                    (SizeT*)(data_slice[0]->keys_marker[0].GetPointer(util::DEVICE)),
                    context[0]);
                //for (peer_ = 0; peer_<num_gpus;peer_++)
                //    util::cpu_mt::PrintGPUArray("keys_marker1", data_slice[0]->keys_marker[peer_].GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration, -1, stream);

                if (graph_slice->nodes > 0)
                {
                    cudaMemcpyAsync(
                        &data_slice[0]->out_length[peer_],
                        data_slice[0]->keys_marker[0].GetPointer(util::DEVICE) 
                            + (graph_slice->nodes -1),
                        sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
                } else {
                    if (peer_ > 0)
                        data_slice[0]->out_length[peer_] = 0;
                }
                if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;

                bool over_sized = false;
                if (peer_>1) {
                    data_slice[0]->keys_out[peer_] = data_slice[0]->temp_keys_out[peer_];
                    data_slice[0]->temp_keys_out[peer_] = util::Array1D<SizeT, VertexId>();
                }
                if (enactor_stats->retval = Check_Size<SizeT, VertexId> (
                    enactor -> size_check, "keys_out", 
                    data_slice[0] -> out_length[peer_], 
                   &data_slice[0] -> keys_out[peer_], 
                    over_sized, thread_num, enactor_stats->iteration, peer_)) 
                    return;
                if (peer_>0)
                    if (enactor_stats->retval = Check_Size<SizeT, Value> (
                        enactor -> size_check, "values_out", 
                        data_slice[0]->out_length[peer_], 
                        &data_slice[0]->value__associate_out[peer_][0], 
                        over_sized, thread_num, enactor_stats->iteration, peer_)) 
                        return;
                data_slice[0]->keys_outs[peer_] = data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE);
                //if (!over_sized) continue;
                data_slice[0]->value__associate_outs[peer_][0] 
                    = data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE);
                data_slice[0]->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, -1, 0, stream);

                Assign_Keys_PR <VertexId, SizeT>
                    <<<grid_size, block_size, 0, stream>>> (
                    graph_slice->nodes,
                    peer_,
                    graph_slice->partition_table  .GetPointer(util::DEVICE),
                    data_slice[0]->markers        .GetPointer(util::DEVICE),
                    data_slice[0]->keys_marker [0].GetPointer(util::DEVICE),
                    data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE));
            }
            data_slice[0]->keys_outs.Move(util::HOST, util::DEVICE, -1, 0, stream);
            data_slice[0]->out_length[0] = temp_length;

            //util::cpu_mt::PrintCPUArray("out_length", &data_slice[0]->out_length[0], num_gpus, thread_num, enactor_stats->iteration);
            //for (peer_ = 0; peer_<num_gpus; peer_++)
            //    util::cpu_mt::PrintGPUArray("keys_out[]", data_slice[0]->keys_out[peer_].GetPointer(util::DEVICE), data_slice[0]->out_length[peer_], thread_num, enactor_stats->iteration, peer_, stream);
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
        //for (peer_ = 1; peer_ < num_gpus; peer_++)
        //{
        //    util::cpu_mt::PrintGPUArray("values_out[]", data_slice[0]->value__associate_out[peer_][0].GetPointer(util::DEVICE), data_slice[0]->out_length[peer_], thread_num, enactor_stats->iteration, peer_, stream);
        //}
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
    }

};

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam PrEnactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
static CUT_THREADPROC PRThread(
    void * thread_data_)
{
    typedef typename Enactor::Problem    Problem;
    typedef typename Enactor::SizeT      SizeT;
    typedef typename Enactor::VertexId   VertexId;
    typedef typename Enactor::Value      Value;
    typedef typename Problem::DataSlice  DataSlice;
    typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
    typedef PRFunctor <VertexId, SizeT, Value, Problem> Functor;
    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data->problem;
    Enactor      *enactor            =  (Enactor*)     thread_data->enactor;
    //util::cpu_mt::CPUBarrier
    //             *cpu_barrier        =   thread_data -> cpu_barrier;
    int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
    DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    GraphSliceT  *graph_slice        =   problem     -> graph_slices       [thread_num] ;
    FrontierAttribute<SizeT>
                 *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    EnactorStats *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

    do {
        // printf("PRThread entered\n");fflush(stdout);
        if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
        int *markers = new int [num_gpus];
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
        //gunrock::app::Iteration_Loop
        //    <0, 0, PrEnactor, PrFunctor, R0DIteration<AdvanceKernelPolicy, FilterKernelPolicy, PrEnactor> > (thread_data);

        data_slice->PR_queue_selector = frontier_attribute[0].selector;
        //for (int peer_=0; peer_<num_gpus; peer_++)
        //{
        //    frontier_attribute[peer_].queue_reset = true;
        //    enactor_stats     [peer_].iteration   = 0;
        //}
        if (num_gpus > 1)
        {
            data_slice->value__associate_orgs[0] = data_slice->rank_next.GetPointer(util::DEVICE);
            data_slice->value__associate_orgs.Move(util::HOST, util::DEVICE);
            //util::cpu_mt::IncrementnWaitBarrier(cpu_barrier, thread_num);
            //for (int i=0; i<4; i++)
            //for (int gpu=0; gpu<num_gpus; gpu++)
            //for (int stage=0; stage<data_slice->num_stages; stage++)
            //    data_slice->events_set[i][gpu][stage] = false;
            //util::cpu_mt::IncrementnWaitBarrier(cpu_barrier+1, thread_num);
        }
        data_slice -> edge_map_queue_len = frontier_attribute[0].queue_length;
        //util::cpu_mt::PrintGPUArray("degrees", data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes, thread_num);

        // Step through PR iterations
        gunrock::app::Iteration_Loop
            <Enactor, Functor, 
            PRIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>, 
            0, 1 > (thread_data);

        if (thread_num > 0)
        {
            bool over_sized = false;
            if (enactor_stats -> retval = Check_Size<SizeT, Value>(
                enactor -> size_check, "values_out", 
                data_slice -> local_nodes, &data_slice -> value__associate_out[1][0], 
                over_sized, thread_num, enactor_stats -> iteration, -1)) break;
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "keys_out", 
                data_slice -> local_nodes, &data_slice -> keys_out[1], 
                over_sized, thread_num, enactor_stats -> iteration, -1)) break;
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "keys_out", 
                data_slice -> local_nodes, &data_slice -> keys_out[0],
                over_sized, thread_num, enactor_stats -> iteration, -1)) break;
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
            PushNeighbor <Enactor, GraphSliceT, DataSlice, 0, 1> (
                enactor,
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

            // release some space on GPU, will be allocated again during Reset
            data_slice -> frontier_queues[0].keys[0].Release();
            data_slice -> frontier_queues[0].keys[1].Release();
            data_slice -> rank_next.Release();
            data_slice -> degrees.Release();
            if (enactor_stats -> retval = data_slice->node_ids.Allocate(graph_slice->nodes, util::DEVICE))
                break;
            util::MemsetIdxKernel<<<128, 128>>>(
                data_slice-> node_ids.GetPointer(util::DEVICE), graph_slice->nodes);

            // sort according to the rank of nodes
            util::CUBRadixSort<Value, VertexId>(
                false, graph_slice->nodes,
                data_slice->rank_curr.GetPointer(util::DEVICE),
                data_slice->node_ids.GetPointer(util::DEVICE));

            if (problem -> scaled)
            {
                util::MemsetScaleKernel<<<128, 128>>>(
                    data_slice->rank_curr.GetPointer(util::DEVICE),
                    (Value) (1.0 / (Value) (problem->org_graph->nodes)),
                    graph_slice -> nodes);
            }
        }

    } while(0);

    // printf("PR_Thread finished\n");fflush(stdout);
    thread_data->stats = 4;
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
template <
    typename _Problem>
    //bool _INSTRUMENT,
    //bool _DEBUG,
    //bool _SIZE_CHECK>
class PREnactor :
    public EnactorBase<typename _Problem::SizeT>
{
    // Members
    ThreadSlice *thread_slices;
    CUTThread   *thread_Ids   ;
    util::cpu_mt::CPUBarrier *cpu_barrier;

    // Methods
public:
    _Problem    *problem      ;
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    //static const bool INSTRUMENT = _INSTRUMENT;
    //static const bool DEBUG      = _DEBUG;
    //static const bool SIZE_CHECK = _SIZE_CHECK;

   /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief PREnactor constructor
     */
    PREnactor(
        int   num_gpus   = 1, 
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(VERTEX_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        thread_slices (NULL),
        thread_Ids    (NULL),
        problem       (NULL),
        cpu_barrier   (NULL)
    {
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
        if (cpu_barrier!=NULL)
        {
            util::cpu_mt::DestoryBarrier(&cpu_barrier[0]);
            util::cpu_mt::DestoryBarrier(&cpu_barrier[1]);
            delete[] cpu_barrier;cpu_barrier=NULL;
        }
    }
    
    /** @} */

    /**
     * \addtogroup PublicInterface
     * @{
     */

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
    cudaError_t InitPR(
        ContextPtr  *context,
        Problem     *problem,
        //int         max_iteration,
        int         max_grid_size = 512)
    {
        cudaError_t retval = cudaSuccess;
        cpu_barrier = new util::cpu_mt::CPUBarrier[2];
        cpu_barrier[0]=util::cpu_mt::CreateBarrier(this->num_gpus);
        cpu_barrier[1]=util::cpu_mt::CreateBarrier(this->num_gpus);
        // Lazy initialization
        if (retval = BaseEnactor::Init(
            //problem,
            max_grid_size,
            AdvanceKernelPolity::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        if (this -> debug) {
            printf("PR vertex map occupancy %d, level-grid size %d\n",
                FilterKernelPolicy::CTA_OCCUPANCY, this->enactor_stats[0].filter_grid_size);
        }

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
                (CUT_THREADROUTINE)&(PRThread<
                    AdvanceKernelPolity, FilterKernelPolicy,
                    PREnactor<Problem> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
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
        return BaseEnactor::Reset();
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
            {
                if (this->enactor_stats[gpu].retval!=cudaSuccess)
                {
                    retval=this->enactor_stats[gpu].retval;
                    break;
                }
            }
        } while (0);
        if (this -> debug) printf("\nGPU PR Done.\n");
        return retval;
    }

    /** @} */

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        1,                                  // MIN_CTA_OCCUPANCY
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
        //INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
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
    LBAdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        1,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    FWDAdvanceKernelPolicy;

    /**
     * @brief PageRank Enact kernel entry.
     *
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Enact(
        int   traversal_mode)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
            if (min_sm_version == -1 || this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;

        if (min_sm_version >= 300) {
            if (traversal_mode == 1)
            {
                return EnactPR<FWDAdvanceKernelPolicy, FilterKernelPolicy>();
            }
            else
            {
                return EnactPR< LBAdvanceKernelPolicy, FilterKernelPolicy>();
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }


    /**
     * @brief PageRank Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr *context,
        Problem    *problem,
        int         traversal_mode,
        //int       max_iteration,
        int         max_grid_size = 512)
    {
        int min_sm_version = -1;
        for (int gpu=0; gpu<this->num_gpus; gpu++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[gpu].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[gpu].device_sm_version;
        }

        if (min_sm_version >= 300)
        {
            if (traversal_mode == 1)
                return InitPR<FWDAdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, /*max_iteration,*/ max_grid_size);
            else return InitPR<LBAdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, /*max_iteration,*/ max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all architectures
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
