// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once
#include <limits>
#include <thread>
#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <moderngpu.cuh>

//not sure how to propagate a command line option
//down to FullQueueCore, use an ugly macro for now.

using namespace gunrock::util;

namespace gunrock {
namespace app {
namespace bfs {

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
__global__ void Expand_Incoming_BFS (
    const SizeT            num_elements,
    const VertexId*  const keys_in,
          VertexId*        keys_out,
    const size_t           array_size,
          char*            array,
          int              gpu_idx,
          VertexId         label,
          VertexId         *d_labels)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    size_t offset = 0;
    VertexId** s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
    offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
    VertexId** s_vertex_associate_org = (VertexId**)&(s_array[offset]);
    SizeT x = threadIdx.x;
    while (x < array_size)
    {
        s_array[x] = array[x];
        x += blockDim.x;
    }
    __syncthreads();

    VertexId key,t;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        key = keys_in[x];
        //t   = s_vertex_associate_in[0][x];

        //if (atomicCAS(s_vertex_associate_org[0]+key, (VertexId)-1, t)!= -1)
        //{
           if (atomicMin(d_labels+key, label)<=label)
           //if (s_vertex_associate_org[0][key] <= t)
           {
               keys_out[x]=util::InvalidValue<VertexId>();
               x+=STRIDE;
               continue;
           }
        //}
        keys_out[x]=key;
        //if (util::to_track(gpu_idx, key))
        //    printf("%d\t %s\t labels[%d] -> %d\n",
        //        gpu_idx, __func__, key, t);
        if (NUM_VERTEX_ASSOCIATES == 1 && d_labels[key] == t)
            s_vertex_associate_org[0][key] = s_vertex_associate_in[0][x];
        x+=STRIDE;
    }
}

template <
    typename VertexId,
    typename SizeT>
struct LoadLabel {};

template <typename SizeT>
struct LoadLabel<int, SizeT>
{
    static __device__ __forceinline__ int Load
        (int *&d_labels, SizeT &pos)
    {
        //return tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<int>::labels, pos);
        return _ldg(d_labels + pos);
    }
};

template <typename SizeT>
struct LoadLabel<long long, SizeT>
{
    static __device__ __forceinline__ long long Load
        (long long *&d_labels, SizeT &pos)
    {
        return _ldg(d_labels + pos);
    }
};

template <
    typename KernelPolicy,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
__global__ void Expand_Incoming_Kernel(
                                 int             thread_num,
          typename KernelPolicy::VertexId        label,
          typename KernelPolicy::SizeT           num_elements,
          typename KernelPolicy::VertexId*       d_keys_in,
          typename KernelPolicy::VertexId*       d_vertex_associate_in,
    //const Value*    const d_value__associate_out
          typename KernelPolicy::SizeT*          d_out_length,
          typename KernelPolicy::VertexId*       d_keys_out,
          typename KernelPolicy::VertexId**      d_vertex_associate_orgs,
    //util::Array1D<SizeT, Value*   > value__associate_org,
          typename KernelPolicy::VertexId*       d_labels,
          typename KernelPolicy::Problem::MaskT*          d_masks)
          //typename KernelPolicy::Problem::SizeT* d_row_offsets)
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    VertexId key = util::InvalidValue<VertexId>();
    while (x - threadIdx.x < num_elements)
    {
        bool to_process = true;
        SizeT output_pos = util::InvalidValue<SizeT>();
        SizeT mask_pos = util::InvalidValue<SizeT>();
        MaskT tex_mask_byte ;
        //MaskT tex_mask_byte;
        if (x < num_elements)
        {
            key = _ldg(d_keys_in + x);
            //printf("(%d, %d) : Expand_Incoming, input_keys[%d] = %d\n",
            //    blockIdx.x, threadIdx.x, x, key);
            if (KernelPolicy::Problem::ENABLE_IDEMPOTENCE)
            {
                mask_pos = (key & KernelPolicy::LOAD_BALANCED_CULL::ELEMENT_ID_MASK) >> (2+sizeof(MaskT));
                //tex_mask_byte = tex1Dfetch(
                //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,
                //    mask_pos);
                tex_mask_byte = _ldg(d_masks + mask_pos);
                MaskT mask_bit = 1 << (key & ((1 << (2 + sizeof(MaskT)))-1));
                if (!(mask_bit & tex_mask_byte))
                {
                    tex_mask_byte |= mask_bit;
                } else to_process = false;
            }

            if (to_process)
            {
                VertexId loaded_label = LoadLabel<VertexId, VertexId>::Load(d_labels, key);
                //printf("(%d, %d) : labels[%d] = %d\n",
                //    blockIdx.x, threadIdx.x, key, loaded_label);
                if (loaded_label != util::MaxValue<VertexId>())
                //if (LoadLabel<VertexId, VertexId>::Load(d_labels, key) != util::MaxValue<VertexId>())
                        to_process = false;
            }

            if (to_process)
            {
                d_labels[key] = label;
                if (KernelPolicy::Problem::ENABLE_IDEMPOTENCE)
                    d_masks[mask_pos] = tex_mask_byte;
            }

            //if (to_process)
            //{
            //    if (tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,  key + 1) == tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,  key))
                //if (_ldg(d_row_offsets + key) == _ldg(d_row_offsets + (key+1)))
            //        to_process = false;
            //}
        } else to_process = false;

        BlockScanT::LogicScan(to_process, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            if (output_pos != 0 || to_process)
            block_offset = atomicAdd(d_out_length, output_pos + ((to_process) ? 1 : 0));
        }
        __syncthreads();


        if (to_process)
        {
            //printf("(%4d, %4d) : in_pos = %d, key = %d, out_pos = %d + %d\n",
            //    blockIdx.x, threadIdx.x, x, key, output_pos, block_offset);
            output_pos += block_offset;
            d_keys_out[output_pos] = key;
            if (NUM_VERTEX_ASSOCIATES != 0 && d_labels[key] == label)
                d_vertex_associate_orgs[0][key] = d_vertex_associate_in[x];
        }
        x += STRIDE;
    }
}

template <typename Problem, typename KernelPolicy>
__global__ void From_Unvisited_Queue(
    typename Problem::SizeT     num_nodes,
    typename Problem::SizeT    *d_out_length,
    typename Problem::VertexId *d_keys_out,
    typename Problem::VertexId *d_labels)
{
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;

    while (x - threadIdx.x < num_nodes)
    {
        bool to_process = true;
        SizeT output_pos = 0;
        if (x < num_nodes)
        {
            //printf("(%d, %d) : x = %d\n", blockIdx.x, threadIdx.x, x);
            //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels, x)
            //    != util::MaxValue<VertexId>())
            if (LoadLabel<VertexId, SizeT>::Load(d_labels, x) != util::MaxValue<VertexId>())
                to_process = false;
        } else to_process = false;

        BlockScanT::LogicScan(to_process, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            block_offset = atomicAdd(d_out_length, output_pos + ((to_process) ? 1 : 0));
        }
        __syncthreads();
        if (to_process)
        {
            output_pos += block_offset;
            d_keys_out[output_pos] = x;
        }
        __syncthreads();
        x += STRIDE;
    }
}

template <typename Problem, typename KernelPolicy>
__global__ void From_Unvisited_Queue_IDEM(
    typename Problem::SizeT     num_nodes,
    typename Problem::SizeT    *d_out_length,
    typename Problem::VertexId *d_keys_out,
    typename Problem::MaskT    *d_visited_mask,
    typename Problem::SizeT    *d_row_offsets,
    typename Problem::VertexId *d_labels)
{
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    VertexId l_keys_out[sizeof(MaskT) * 8];
    SizeT    l_counter = 0;
    SizeT    output_pos = 0;

    while ((x - threadIdx.x) * sizeof(MaskT) * 8 < num_nodes)
    {
        MaskT mask_byte = 0;
        bool changed = false;
        l_counter = 0;
        if (x * (sizeof(MaskT) << 3) < num_nodes)
        {
            //mask_byte = tex1Dfetch(
            //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,
            //    x);
            mask_byte = _ldg(d_visited_mask + x);
            if (mask_byte != util::AllOnes<MaskT>())
            for (int i=0; i<(1 << (2+sizeof(MaskT))); i++)
            {
                MaskT mask_bit = 1 << i;
                VertexId key = 0;
                //bool to_process = true;
                if (mask_byte & mask_bit) continue;

                key = (x << (sizeof(MaskT) + 2)) + i;
                if (key >= num_nodes) break;

                //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels, key) != util::MaxValue<VertexId>())
                if (_ldg(d_labels + key) != util::MaxValue<VertexId>())
                {
                    //to_process = false;
                    mask_byte |= mask_bit;
                    changed = true;
                    continue;
                }
                //if (to_process)
                { // only works for undirected graph
                    //if (tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets, key) == tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets, key+1)) to_process = false;
                    //if (_ldg(d_row_offsets + key) == _ldg(d_row_offsets + (key+1)))
                    //    continue;
                }
                //if (to_process)
                {
                    l_keys_out[l_counter] = key;
                    l_counter ++;
                }
            }
            if (changed)
                d_visited_mask[x] = mask_byte;
        }

        BlockScanT::Scan(l_counter, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            block_offset = atomicAdd(d_out_length, output_pos + l_counter);
        }
        __syncthreads();
        output_pos += block_offset;
        for (int i=0; i<l_counter; i++)
        {
            d_keys_out[output_pos] = l_keys_out[i];
            output_pos ++;
        }
        __syncthreads();
        x += STRIDE;
    }
}

template <typename Problem, typename KernelPolicy>
__global__ void From_Unvisited_Queue_Local(
    typename Problem::SizeT     num_local_vertices,
    typename Problem::VertexId *d_local_vertices,
    typename Problem::SizeT    *d_out_length,
    typename Problem::VertexId *d_keys_out,
    typename Problem::VertexId *d_labels)
{
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;

    while ((x - threadIdx.x) < num_local_vertices)
    {
        bool to_process = true;
        VertexId key = 0;
        SizeT output_pos = 0;

        if (x < num_local_vertices)
        {
            key = d_local_vertices[x];
            //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels, key)
            if (LoadLabel<VertexId, VertexId>::Load(d_labels, key)
                != util::MaxValue<VertexId>())
                to_process = false;
        } else to_process = false;

        BlockScanT::LogicScan(to_process, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            block_offset = atomicAdd(d_out_length, output_pos + ((to_process) ? 1 : 0));
        }
        __syncthreads();
        if (to_process)
        {
            output_pos += block_offset;
            d_keys_out[output_pos] = key;
        }
        __syncthreads();
        x += STRIDE;
    }
}

template <typename Problem, typename KernelPolicy>
__global__ void From_Unvisited_Queue_Local_IDEM(
    typename Problem::SizeT     num_local_vertices,
    typename Problem::VertexId *d_local_vertices,
    typename Problem::SizeT    *d_out_length,
    typename Problem::VertexId *d_keys_out,
    typename Problem::MaskT    *d_visited_mask,
    typename Problem::VertexId *d_labels)
{
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT) blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    //VertexId l_keys_out[sizeof(MaskT) * 8];
    //SizeT    l_counter = 0;
    //SizeT    output_pos = 0;

    //while ((x - threadIdx.x) * sizeof(MaskT) * 8 < num_nodes)
    while ((x-threadIdx.x) < num_local_vertices)
    {
        //MaskT mask_byte = 0;
        bool to_process = true;
        VertexId key = 0;
        SizeT output_pos = 0;
        if (x < num_local_vertices)
        {
            key = d_local_vertices[x];
            SizeT mask_pos = (key & KernelPolicy::LOAD_BALANCED_CULL::ELEMENT_ID_MASK) >> (2+sizeof(MaskT));
            //MaskT mask_byte = tex1Dfetch(
            //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,
            //    mask_pos);
            MaskT mask_byte = _ldg(d_visited_mask + mask_pos);
            MaskT mask_bit = 1 << (key & ((1 << (2 + sizeof(MaskT)))-1));
            if (mask_byte & mask_bit)
                to_process = false;
            else {
                //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels, key) != util::MaxValue<VertexId>())
                if (LoadLabel<VertexId, VertexId>::Load(d_labels, key) != util::MaxValue<VertexId>())
                {
                    mask_byte |= mask_bit;
                    d_visited_mask[mask_pos] = mask_byte;
                    to_process = false;
                }
            }
        } else to_process = false;

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
        __syncthreads();
        x += STRIDE;
    }
}

template <typename Problem, typename KernelPolicy>
__global__ void Inverse_Expand(
    typename Problem::SizeT     num_unvisited_vertices,
    typename Problem::VertexId  label,
    typename Problem::VertexId *d_unvisited_key_in,
    typename Problem::SizeT    *d_inverse_row_offsets,
    typename Problem::VertexId *d_inverse_column_indices,
    typename Problem::SizeT    *d_split_lengths,
    typename Problem::VertexId *d_unvisited_key_out,
    typename Problem::VertexId *d_visited_key_out,
    typename Problem::MaskT    *d_visited_mask,
    typename Problem::VertexId *d_labels,
    typename Problem::VertexId *d_preds)
{
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

    while (x - threadIdx.x < num_unvisited_vertices)
    {
        VertexId key = 0, parent = 0;
        bool discoverable = false;
        bool to_process = true;
        MaskT mask_byte, mask_bit;
        SizeT mask_pos;

        if (x < num_unvisited_vertices)
        {
            key = _ldg(d_unvisited_key_in + x);
        } else to_process = false;

        if (to_process && Problem::ENABLE_IDEMPOTENCE)
        {
            mask_pos = (key & KernelPolicy::LOAD_BALANCED_CULL::ELEMENT_ID_MASK) >> (2+sizeof(MaskT));
            //mask_byte = tex1Dfetch(
            //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,
            //    mask_pos);
            mask_byte = _ldg(d_visited_mask + mask_pos);
            mask_bit = 1 << (key & ((1 << (2 + sizeof(MaskT)))-1));
            if (mask_byte & mask_bit)
                to_process = false;
        }

        if (to_process)// && !Problem::ENABLE_IDEMPOTENCE)
        {
            //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels, key) != util::MaxValue<VertexId>())
            if (LoadLabel<VertexId, VertexId>::Load(d_labels, key) != util::MaxValue<VertexId>())
            {
                if (Problem::ENABLE_IDEMPOTENCE)
                {
                    mask_byte |= mask_bit;
                    d_visited_mask[mask_pos] = mask_byte;
                }
                to_process = false;
            }
        }

        if (to_process)
        {
            SizeT edge_start = //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets, key);
                _ldg(d_inverse_row_offsets + key);
            SizeT edge_end = //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets, key+1);
                _ldg(d_inverse_row_offsets + (key+1));
            for (SizeT edge_id = edge_start; edge_id < edge_end; edge_id++)
            {
                VertexId neighbor = d_inverse_column_indices[edge_id];
                //if (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels, neighbor) == label-1)
                //printf("%lld <- %lld, edge_id = %lld\n",
                //    (long long)key, (long long)neighbor, (long long)edge_id);
                if (LoadLabel<VertexId, VertexId>::Load(d_labels, neighbor) == label -1)
                {
                    discoverable = true;
                    parent = neighbor;
                    //printf("Found %lld from %lld, label = %lld\n",
                    //(long long)key, (long long)neighbor, (long long)label);
                    break;
                }
            }

        }

        if (discoverable)
        {
            if (Problem::ENABLE_IDEMPOTENCE)
            {
                mask_byte |= mask_bit;
                d_visited_mask[mask_pos] = mask_byte;
            }
            d_labels[key] = label;
            if (Problem::MARK_PREDECESSORS)
                d_preds [key] = parent;
        }

        SizeT output_pos = 0;
        BlockScanT::LogicScan(discoverable, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            if (output_pos != 0 || discoverable)
            block_offset = atomicAdd(d_split_lengths +1, output_pos + ((discoverable) ? 1 : 0));
        }
        __syncthreads();
        if (discoverable && d_visited_key_out != NULL)
        {
            output_pos += block_offset;
            d_visited_key_out[output_pos] = key;
        }
        __syncthreads();

        to_process = to_process && (!discoverable);
        BlockScanT::LogicScan(to_process, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            if (output_pos != 0 || to_process)
            block_offset = atomicAdd(d_split_lengths, output_pos + ((to_process) ? 1 : 0));
        }
        __syncthreads();
        if (to_process && d_unvisited_key_out != NULL)
        {
            output_pos += block_offset;
            d_unvisited_key_out[output_pos] = key;
        }
        __syncthreads();

        x += STRIDE;
    }
}

template <typename Problem>
__global__ void Update_Mask_Kernel(
    typename Problem::SizeT num_nodes,
    typename Problem::MaskT* old_masks,
    typename Problem::MaskT*       visited_masks,
    typename Problem::VertexId* labels)
{
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::MaskT    MaskT;

    SizeT end_mask_pos = (num_nodes >> 3) / sizeof(MaskT) + 1;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

    while ( x < end_mask_pos)
    {
        MaskT mask = _ldg(visited_masks + x);
        VertexId v = (x << 3) * sizeof(MaskT);
        bool has_change = false;
        #pragma unroll
        for (int i=0; i< (sizeof(MaskT) << 3); i++)
        {
            MaskT mask_bit = 1 << i;
            if ((!(mask & mask_bit)) && (v < num_nodes))
            {
                if (_ldg(labels + v) != util::MaxValue<VertexId>())
                {
                    mask |= mask_bit;
                    has_change = true;
                    //printf("mask missed %d\n", v);
                }
            }
            v++;
        }
        if (has_change) visited_masks[x] = mask;
        x += STRIDE;
    }
}

template <typename Problem, typename KernelPolicy>
__global__ void Combind_Masks(
    typename Problem::SizeT  num_nodes,
    int                      num_gpus,
    typename Problem::VertexId label,
    typename Problem::MaskT *d_old_masks,
    typename Problem::MaskT *d_visited_masks,
    typename Problem::MaskT **d_mask_ins,
    typename Problem::VertexId *d_out_keys,
    typename Problem::SizeT *d_out_key_length,
    typename Problem::VertexId *d_labels,
    int *d_partition_table)
{
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::MaskT    MaskT;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    __shared__ MaskT* s_mask_ins[8];
    SizeT end_mask_pos = num_nodes / 8 / sizeof(MaskT) + 1;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    int l_vertex_counter = 0;
    VertexId l_vertices[sizeof(MaskT) * 8];

    if (threadIdx.x < num_gpus) s_mask_ins[threadIdx.x] = d_mask_ins[threadIdx.x];
    __syncthreads();

    while (x - threadIdx.x < end_mask_pos)
    {
        l_vertex_counter = 0;
        if (x < end_mask_pos)
        {
            MaskT new_mask = d_visited_masks[x];
            MaskT local_mask = new_mask;
            for (int gpu = 1; gpu < num_gpus; gpu++)
            {
                MaskT in_mask = 0;
                if (d_out_key_length[gpu] != 0)
                    in_mask = _ldg(s_mask_ins[gpu] + x);
                //printf("(%d, %d) : in_mask = %#x\n", blockIdx.x, threadIdx.x, in_mask);
                new_mask |= in_mask;
            }
            MaskT old_mask = d_old_masks[x];
            MaskT update_mask = (~old_mask) & new_mask;
            //printf("(%d, %d) : new_mask = %#x, local_mask = %#x, old_mask = %#x, update_mask = %#x, label = %d\n",
            //    blockIdx.x, threadIdx.x, new_mask, local_mask, old_mask, update_mask, label);
            if (update_mask != 0)
            {
                VertexId v = (x << 3) * sizeof(MaskT);
                for (int i=0; i<(sizeof(MaskT) << 3); i++)
                {
                    MaskT mask_bit = 1 << i;
                    if ((update_mask & mask_bit) && (v < num_nodes))
                    {
                        if (_ldg(d_partition_table + v) == 0)
                        {
                            l_vertices[l_vertex_counter] = v;
                            l_vertex_counter ++;
                        }
                        if (!(local_mask & mask_bit))
                            d_labels[v] = label;
                    }
                    v++;
                }
            }
            d_old_masks[x] = new_mask;
            d_visited_masks[x] = new_mask;
        }

        SizeT output_pos = 0;
        BlockScanT::Scan(l_vertex_counter, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            if (output_pos + l_vertex_counter != 0)
                block_offset = atomicAdd(d_out_key_length, output_pos + l_vertex_counter);
        }
        __syncthreads();
        if (l_vertex_counter != 0)
        {
            output_pos += block_offset;
            for (int i=0; i< l_vertex_counter; i++)
            {
                d_out_keys[output_pos + i] = l_vertices[i];
            }
        }
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
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct BFSIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    //true, false, false, true, Enactor::Problem::MARK_PREDECESSORS>
    false, true, false, true, Enactor::Problem::MARK_PREDECESSORS>
{
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef typename Problem::MaskT      MaskT     ;
    //typedef typename Enactor::Frontier   Frontier  ;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value>
                                        Frontier;
    typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
    typedef BFSFunctor<VertexId, SizeT, Value, Problem> Functor;
    typedef IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        false, true, false, true, Enactor::Problem::MARK_PREDECESSORS>
        BaseIteration;

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
        if (false)//(Problem::ENABLE_IDEMPOTENCE && (!Problem::MARK_PREDECESSORS) &&
           // (data_slice -> num_gpus > 1) && (enactor_stats -> iteration > 0))
        {
            for (int gpu = 1; gpu < data_slice -> num_gpus; gpu++)
            {
                data_slice -> in_masks[gpu] =
                    (MaskT*)(data_slice -> keys_in[enactor_stats -> iteration % 2][gpu].GetPointer(util::DEVICE));
                //printf("%d\t \t %d, in_mask = %p\n",
                //    thread_num, gpu, data_slice -> in_masks[gpu]);
            }
            data_slice -> in_masks.Move(util::HOST, util::DEVICE, data_slice -> num_gpus, 0, stream);
            data_slice -> in_length_out[0] = 0;
            data_slice -> in_length_out.Move(util::HOST, util::DEVICE, data_slice -> num_gpus, 0, stream);

            int num_blocks = graph_slice -> nodes / 32 / sizeof(MaskT) / AdvanceKernelPolicy::THREADS + 1;
            if (num_blocks > 480) num_blocks = 480;
            Combind_Masks<Problem, AdvanceKernelPolicy>
                <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>(
                graph_slice -> nodes,
                data_slice -> num_gpus,
                (VertexId)enactor_stats -> iteration,
                data_slice -> old_mask.GetPointer(util::DEVICE),
                data_slice -> visited_mask.GetPointer(util::DEVICE),
                data_slice -> in_masks.GetPointer(util::DEVICE),
                frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                data_slice -> in_length_out.GetPointer(util::DEVICE),
                data_slice -> labels.GetPointer(util::DEVICE),
                graph_slice -> partition_table.GetPointer(util::DEVICE));
            data_slice -> in_length_out.Move(util::DEVICE, util::HOST, 1, 0, stream);
            if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize failed", __FILE__, __LINE__))
                return;
            frontier_attribute -> queue_length = data_slice -> in_length_out[0];
        }

         //if (data_slice -> previous_direction == FORWARD)
            data_slice -> num_visited_vertices += frontier_attribute -> queue_length;
        data_slice -> num_unvisited_vertices = graph_slice -> nodes - data_slice -> num_visited_vertices;
        float predicted_backward_visits = data_slice -> num_visited_vertices == 0 ?
            std::numeric_limits<float>::infinity() :
            data_slice -> num_unvisited_vertices * 1.0 * graph_slice -> nodes / data_slice -> num_visited_vertices;
        //predicted_reverse_edge_visits /= enactor -> num_gpus;
        float predicted_forward_visits = frontier_attribute -> queue_length * 1.0 * graph_slice -> edges / graph_slice -> nodes;
        //printf("%d\t %d\t %d\t queue_length = %d, output_length = %d, visited = %d, rev = %f\n",
        //    data_slice -> gpu_idx, enactor_stats -> iteration, peer_,
        //    frontier_attribute -> queue_length,
        //    frontier_attribute -> output_length[0],
        //    data_slice -> num_visited_vertices,
        //    rev);
        long long iteration_ = enactor_stats -> iteration % 4;
        if (enactor -> direction_optimized)
        {
            if (data_slice -> previous_direction == FORWARD)
            {
                if (predicted_forward_visits > predicted_backward_visits * enactor -> do_a &&
                    !data_slice -> been_in_backward)
                     data_slice -> direction_votes[iteration_] = BACKWARD;
                else data_slice -> direction_votes[iteration_] = FORWARD;
            } else {
                data_slice -> been_in_backward = true;
                if (predicted_forward_visits > predicted_backward_visits * enactor -> do_b)
                     data_slice -> direction_votes[iteration_] = BACKWARD;
                else data_slice -> direction_votes[iteration_] = FORWARD;
            }
        } else data_slice -> direction_votes[iteration_] = FORWARD;
        data_slice -> direction_votes[(iteration_+1)%4] = UNDECIDED;

        if (enactor -> num_gpus > 1 && enactor_stats -> iteration != 0 && enactor -> direction_optimized)
        {
            /*int vote_counter[3];
            data_slice -> current_direction = UNDECIDED;
            while (data_slice -> current_direction == UNDECIDED)
            {
                for (int i=0; i<3; i++) vote_counter[i] = 0;
                for (int peer = 0; peer < enactor -> num_gpus; peer++)
                    vote_counter[enactor -> problem -> data_slices[peer] -> direction_votes[iteration_]]++;
                //printf("%d\t %d\t counts : %d, %d, %d\n",
                //    thread_num, enactor_stats -> iteration,
                //    vote_counter[0], vote_counter[1], vote_counter[2]);
                if (vote_counter[FORWARD] > enactor -> num_gpus / 2)
                    data_slice -> current_direction = FORWARD;
                else if (vote_counter[BACKWARD] > enactor -> num_gpus /2)
                    data_slice -> current_direction = BACKWARD;
                else if (vote_counter[UNDECIDED] == 0)
                    data_slice -> current_direction = enactor -> problem -> data_slices[0] -> direction_votes[iteration_];
                else if (vote_counter[FORWARD] + vote_counter[UNDECIDED] <= vote_counter[BACKWARD] && enactor -> problem -> data_slices[0] -> direction_votes[iteration_] == BACKWARD)
                    data_slice -> current_direction = BACKWARD;
                else if (vote_counter[BACKWARD] + vote_counter[UNDECIDED] <= vote_counter[FORWARD] && enactor -> problem -> data_slices[0] -> direction_votes[iteration_] == FORWARD)
                    data_slice -> current_direction = FORWARD;
                //std::this_thread::yield();
                sleep(0);
            }*/
            while (enactor ->problem -> data_slices[0] -> direction_votes[iteration_] == UNDECIDED)
            {
                sleep(0);
                //std::this_thread::yield();
            }
            data_slice -> current_direction = enactor->problem -> data_slices[0] -> direction_votes[iteration_];
        } else if (enactor_stats -> iteration == 0)
            data_slice -> direction_votes[iteration_] = FORWARD;
        else {
            data_slice -> current_direction = data_slice -> direction_votes[iteration_];
        }
        if (false) //(thread_num == 0 && enactor -> direction_optimized)
            printf("%d\t %lld\t \t queue = %lld,\t pre_output = %lld,\t visited = %lld,\t "
                "unvisited = %lld,\t predicted_forward = %.0f,\t predicted_backward = %.0f,\t "
                "direction = %s, ratio = %.8f\n",
                //" ratio2 = %.4f, ratio3 = %.4f, %.4f, ratio4 = %.4f, ratio5 = %.4f\n",
                thread_num, enactor_stats -> iteration,
                (long long)frontier_attribute -> queue_length,
                (long long)frontier_attribute -> output_length[0],
                (long long)data_slice -> num_visited_vertices,
                (long long)data_slice -> num_unvisited_vertices,
                predicted_forward_visits,
                predicted_backward_visits,
                data_slice -> current_direction == FORWARD ? "FORWARD" : "BACKWARD",
                predicted_forward_visits / predicted_backward_visits);
                //data_slice -> num_visited_vertices * 1.0 / data_slice -> nodes,
                //data_slice -> num_visited_vertices * 1.0 / data_slice -> num_unvisited_vertices,
                //data_slice -> num_unvisited_vertices * 1.0 / data_slice -> num_visited_vertices,
                //frontier_attribute -> queue_length * 1.0 / data_slice -> num_unvisited_vertices,
                //frontier_attribute -> queue_length * 1.0 * data_slice -> edges / data_slice -> nodes /
                //(data_slice -> num_unvisited_vertices * 1.0 * data_slice -> nodes / data_slice -> num_visited_vertices));
    }

    /*
     * @brief SubQueue_Core function.
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
        if (TO_TRACK)
        {
            printf("%d\t %lld\t %d SubQueue_Core queue_length = %lld\n",
                thread_num, (long long)enactor_stats->iteration, peer_,
                (long long)frontier_attribute -> queue_length);
            fflush(stdout);
            //util::MemsetKernel<<<256, 256, 0, stream>>>(
            //    frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE),
            //    (VertexId)-2,
            //    frontier_queue -> keys[frontier_attribute -> selector^1].GetSize());
            //util::Check_Exist<<<256, 256, 0, stream>>>(
            //    frontier_attribute -> queue_length,
            //    data_slice->gpu_idx, 2, enactor_stats -> iteration,
            //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));
            //util::Verify_Value<<<256, 256, 0, stream>>>(
            //    data_slice -> gpu_idx, 2, frontier_attribute -> queue_length,
            //    enactor_stats -> iteration,
            //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE),
            //    data_slice -> labels.GetPointer(util::DEVICE),
            //    (Value)(enactor_stats -> iteration));
        }


        //util::MemsetKernel<<<256, 256, 0, stream>>>(
        //    data_slice -> output_counter.GetPointer(util::DEVICE),
        //    0, frontier_attribute -> output_length[0]);
        //util::MemsetKernel<<<256, 256, 0, stream>>>(
        //    data_slice -> input_counter.GetPointer(util::DEVICE),
        //    0, frontier_attribute -> queue_length);
        //util::MemsetKernel<<<256, 256, 0, stream>>>(
        //    data_slice -> edge_marker.GetPointer(util::DEVICE),
        //    0, graph_slice -> edges);

        //util::cpu_mt::PrintGPUArray("AdvanceInput",
        //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
        //    frontier_attribute -> queue_length,
        //    thread_num, enactor_stats -> iteration, -1, stream);

#ifdef RECORD_PER_ITERATION_STATS
        GpuTimer gpu_timer;
#endif
        if (data_slice -> current_direction == FORWARD)
        {
            frontier_attribute->queue_reset = true;
            enactor_stats     ->nodes_queued[0] += frontier_attribute->queue_length;

            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Forward Advance begin",
                    thread_num, enactor_stats->iteration, peer_);
            VertexId *output_key_pointer = frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE);
            if (false)//(gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>() &&
                //data_slice -> num_gpus > 1 && Problem::ENABLE_IDEMPOTENCE && !Problem::MARK_PREDECESSORS)
            {
                output_key_pointer = NULL;
            }

            // Edge Map
#ifdef RECORD_PER_ITERATION_STATS
            gpu_timer.Start();
#endif
            gunrock::oprtr::advance::LaunchKernel
                <AdvanceKernelPolicy, Problem, Functor, gunrock::oprtr::advance::V2V>(
                enactor_stats[0],
                frontier_attribute[0],
                enactor_stats -> iteration + 1,
                data_slice,
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                scanned_edges ->GetPointer(util::DEVICE),
                frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),
                output_key_pointer, //frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (Value*   )NULL,
                frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,
                frontier_queue -> keys[frontier_attribute -> selector^1].GetSize(),
                work_progress[0],
                context[0],
                stream,
                //gunrock::oprtr::advance::V2V,
                false,
                false,
                false);

#ifdef RECORD_PER_ITERATION_STATS
                gpu_timer.Stop();
                float elapsed = gpu_timer.ElapsedMillis();
                float mteps = frontier_attribute->output_length[0] / (elapsed*1000);
                enactor_stats->per_iteration_advance_time.push_back(elapsed);
                enactor_stats->per_iteration_advance_mteps.push_back(mteps);
                enactor_stats->per_iteration_advance_input_edges.push_back(frontier_attribute->queue_length);
                enactor_stats->per_iteration_advance_output_edges.push_back(frontier_attribute->output_length[0]);
                enactor_stats->per_iteration_advance_direction.push_back(true);
#endif

            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Forward Advance end",
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

            //if (enactor_stats->retval = work_progress -> GetQueueLength(
            //    frontier_attribute->queue_index,
            //    frontier_attribute->queue_length,
            //    false,
            //    stream,
            //    true)) return;
            //if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
            //    "cudaStreamSynchronize failed", __FILE__, __LINE__))
            //    return;
            //printf("%d\t %lld\t %lld\t Queue_Length = %lld, array_size = %lld\n",
            //    thread_num, (long long)-1, (long long)enactor_stats -> iteration,
            //    (long long)frontier_attribute -> queue_length,
            //    (long long)frontier_queue -> keys[frontier_attribute -> selector^1].GetSize());
            //fflush(stdout);
            //util::cpu_mt::PrintGPUArray("AdvanceResult",
            //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
            //    frontier_attribute -> queue_length,
            //    thread_num, enactor_stats -> iteration, -1, stream);

            if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
            {
                // Filter
                if (enactor -> debug)
                    util::cpu_mt::PrintMessage("Filter begin.",
                        thread_num, enactor_stats->iteration);

                gunrock::oprtr::filter::LaunchKernel
                    <FilterKernelPolicy, Problem, Functor> (
                    enactor_stats[0],
                    frontier_attribute[0],
                    (VertexId)enactor_stats -> iteration + 1,
                    data_slice,
                    d_data_slice,
                    data_slice -> vertex_markers[enactor_stats -> iteration % 2].GetPointer(util::DEVICE),
                    data_slice -> visited_mask.GetPointer(util::DEVICE),
                    frontier_queue->keys  [frontier_attribute -> selector  ].GetPointer(util::DEVICE),
                    frontier_queue->keys  [frontier_attribute -> selector^1].GetPointer(util::DEVICE),
                    frontier_queue->values[frontier_attribute -> selector  ].GetPointer(util::DEVICE),
                    (Value*)NULL,
                    frontier_attribute -> output_length[0],
                    graph_slice -> nodes,
                    work_progress[0],
                    context[0],
                    stream,
                    frontier_queue -> keys  [frontier_attribute -> selector  ].GetSize(),
                    frontier_queue -> keys  [frontier_attribute -> selector^1].GetSize(),
                    enactor_stats -> filter_kernel_stats,
                    true, // filtering_flag
                    false); // skip_marking
                if (enactor -> debug)
                    util::cpu_mt::PrintMessage("Filter end.",
                        thread_num, enactor_stats->iteration);

                //if (FilterKernelPolicy::FILTER_MODE == gunrock::oprtr::filter::SIMPLIFIED)
                //    util::MemsetKernel<<<256, 256, 0, stream>>>(
                //        data_slice -> vertex_markers[(enactor_stats -> iteration +1)%2].GetPointer(util::DEVICE), (SizeT)0, graph_slice -> nodes + 1);
                //if (enactor -> debug && (enactor_stats->retval =
                //    util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;

                //if (enactor_stats->retval = work_progress -> GetQueueLength(
                //    frontier_attribute->queue_index,
                //    frontier_attribute->queue_length,
                //    false,
                //    stream,
                //    true)) return;
                //if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
                //    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                //    return;
                //util::cpu_mt::PrintGPUArray("FilterResult",
                //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                //    frontier_attribute -> queue_length,
                //    thread_num, enactor_stats -> iteration, -1, stream);

            }

            if (enactor_stats -> retval = work_progress -> GetQueueLength(
                frontier_attribute -> queue_index,
                frontier_attribute -> queue_length,
                false,
                stream,
                true)) return;

        } else { // backward
            SizeT num_blocks = 0;
            if (data_slice -> previous_direction == FORWARD)
            {
                data_slice -> split_lengths[0] = 0;
                data_slice -> split_lengths.Move(util::HOST, util::DEVICE, 1, 0, stream);
                if (enactor -> num_gpus == 1)
                //if (true)
                {
                    if (Problem::ENABLE_IDEMPOTENCE)
                    {
                        num_blocks = (graph_slice -> nodes >> (2 + /*sizeof(Problem::MaskT)*/1))/ AdvanceKernelPolicy::THREADS + 1;
                        if (num_blocks > 480) num_blocks = 480;
                        From_Unvisited_Queue_IDEM<Problem, AdvanceKernelPolicy>
                            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                            (graph_slice -> nodes,
                            data_slice -> split_lengths.GetPointer(util::DEVICE),
                            data_slice -> unvisited_vertices[frontier_attribute -> selector].GetPointer(util::DEVICE),
                            data_slice -> visited_mask.GetPointer(util::DEVICE),
                            graph_slice -> row_offsets.GetPointer(util::DEVICE),
                            data_slice -> labels.GetPointer(util::DEVICE));
                    } else {
                        num_blocks = graph_slice -> nodes / AdvanceKernelPolicy::THREADS + 1;
                        if (num_blocks > 480) num_blocks = 480;
                        From_Unvisited_Queue<Problem, AdvanceKernelPolicy>
                            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                            (graph_slice -> nodes,
                            data_slice -> split_lengths.GetPointer(util::DEVICE),
                            data_slice -> unvisited_vertices[frontier_attribute -> selector].GetPointer(util::DEVICE),
                            data_slice -> labels.GetPointer(util::DEVICE));
                    }
                } else {
                    num_blocks = data_slice -> local_vertices.GetSize() / AdvanceKernelPolicy::THREADS + 1;
                    if (num_blocks > 480) num_blocks = 480;
                    if (Problem::ENABLE_IDEMPOTENCE)
                    {
                        From_Unvisited_Queue_Local_IDEM<Problem, AdvanceKernelPolicy>
                            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                            (data_slice -> local_vertices.GetSize(),
                            data_slice -> local_vertices.GetPointer(util::DEVICE),
                            data_slice -> split_lengths.GetPointer(util::DEVICE),
                            data_slice -> unvisited_vertices[frontier_attribute -> selector].GetPointer(util::DEVICE),
                            data_slice -> visited_mask.GetPointer(util::DEVICE),
                            data_slice -> labels.GetPointer(util::DEVICE));
                    } else {
                        From_Unvisited_Queue_Local<Problem, AdvanceKernelPolicy>
                            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                            (data_slice -> local_vertices.GetSize(),
                            data_slice -> local_vertices.GetPointer(util::DEVICE),
                            data_slice -> split_lengths.GetPointer(util::DEVICE),
                            data_slice -> unvisited_vertices[frontier_attribute -> selector].GetPointer(util::DEVICE),
                            data_slice -> labels.GetPointer(util::DEVICE));
                    }
                }
                data_slice -> split_lengths.Move(util::DEVICE, util::HOST, 1, 0, stream);
                if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                    return;
                data_slice -> num_unvisited_vertices = data_slice -> split_lengths[0];

                data_slice -> num_visited_vertices = graph_slice -> nodes - data_slice -> num_unvisited_vertices;
                //printf("%d\t actual unvisited = %lld\n", thread_num, (long long)num_unvisited_vertices);
            } else {
                data_slice -> num_unvisited_vertices = data_slice -> split_lengths[0];
            }
            //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("unvisited0",
            //    data_slice -> unvisited_vertices[frontier_attribute -> selector].GetPointer(util::DEVICE),
            //    num_unvisited_vertices, data_slice -> gpu_idx, enactor_stats -> iteration, -1, stream);

            data_slice -> split_lengths[0] = 0;
            data_slice -> split_lengths[1] = 0;
            data_slice -> split_lengths.Move(util::HOST, util::DEVICE, 2, 0, stream);
            enactor_stats     ->nodes_queued[0] += data_slice -> num_unvisited_vertices;
            VertexId *output_key_pointer = frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE);
            if (false) //(data_slice -> num_gpus > 1 && Problem::ENABLE_IDEMPOTENCE && !Problem::MARK_PREDECESSORS)
                output_key_pointer = NULL;
            num_blocks = data_slice -> num_unvisited_vertices / AdvanceKernelPolicy::THREADS + 1;
            if (num_blocks > 480) num_blocks = 480;

#ifdef RECORD_PER_ITERATION_STATS
            gpu_timer.Start();
#endif
            Inverse_Expand<Problem, AdvanceKernelPolicy>
                <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
                (data_slice   -> num_unvisited_vertices,
                enactor_stats -> iteration + 1,
                data_slice    -> unvisited_vertices[frontier_attribute -> selector].GetPointer(util::DEVICE),
                (graph_slice  -> column_offsets  .GetPointer(util::DEVICE) != NULL) ?
                    graph_slice -> column_offsets.GetPointer(util::DEVICE) :
                    graph_slice -> row_offsets   .GetPointer(util::DEVICE),
                (graph_slice  -> row_indices     .GetPointer(util::DEVICE) != NULL) ?
                    graph_slice -> row_indices   .GetPointer(util::DEVICE) :
                    graph_slice -> column_indices.GetPointer(util::DEVICE),
                data_slice -> split_lengths      .GetPointer(util::DEVICE),
                data_slice -> unvisited_vertices[frontier_attribute -> selector ^ 1].GetPointer(util::DEVICE),
                output_key_pointer, //frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE),
                data_slice -> visited_mask.GetPointer(util::DEVICE),
                data_slice -> labels.GetPointer(util::DEVICE),
                data_slice -> preds.GetPointer(util::DEVICE));

#ifdef RECORD_PER_ITERATION_STATS
                gpu_timer.Stop();
                float elapsed = gpu_timer.ElapsedMillis();
                enactor_stats->per_iteration_advance_time.push_back(elapsed);
                enactor_stats->per_iteration_advance_mteps.push_back(-1.0f);
                enactor_stats->per_iteration_advance_input_edges.push_back(-1.0f);
                enactor_stats->per_iteration_advance_output_edges.push_back(-1.0f);
                enactor_stats->per_iteration_advance_direction.push_back(false);
#endif

            data_slice -> split_lengths.Move(util::DEVICE, util::HOST, 2, 0, stream);
            if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize failed", __FILE__, __LINE__))
                return;
            //printf("%d\t discovered = %d, unvisited = %d\n",
            //    thread_num,
            //    data_slice -> split_lengths[1], data_slice -> split_lengths[0]);
            frontier_attribute -> queue_length = data_slice -> split_lengths[1];
            data_slice -> num_visited_vertices = data_slice -> nodes - data_slice -> num_unvisited_vertices;
            enactor_stats -> edges_queued[0] += frontier_attribute -> output_length[0];
            frontier_attribute -> queue_reset = false;
            frontier_attribute -> queue_index++;
            frontier_attribute -> selector ^= 1;
        }
        data_slice -> previous_direction = data_slice -> current_direction;
        // Only need to reset queue for once
        //if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
        //    "cudaStreamSynchronize failed", __FILE__, __LINE__))
        //    return;
        //printf("%d, resulted queue_length = %d\n", thread_num, frontier_attribute ->queue_length);
        //if (enactor_stats -> retval = util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PostAdvance",
        //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
        //    frontier_attribute -> queue_length,
        //    thread_num, enactor_stats -> iteration, peer_, stream))
        //    return;

        //if (enactor -> debug)
        //    util::cpu_mt::PrintMessage("Filter begin",
        //        thread_num, enactor_stats->iteration, peer_);
        if (TO_TRACK)
        {
            util::Check_Value<<<1,1,0,stream>>>(
                work_progress -> template GetQueueLengthPointer<unsigned int>(
                    frontier_attribute->queue_index),
                data_slice->gpu_idx, 3, enactor_stats -> iteration);
            //util::Check_Exist_<<<256, 256, 0, stream>>>(
            //    work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
            //        frontier_attribute->queue_index),
            //    data_slice->gpu_idx, 3, enactor_stats -> iteration,
            //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));
            //util::Verify_Value_<<<256, 256, 0, stream>>>(
            //    data_slice -> gpu_idx, 3,
            //    work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
            //        frontier_attribute -> queue_index),
            //    enactor_stats -> iteration,
            //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE),
            //    data_slice -> labels.GetPointer(util::DEVICE),
            //    enactor_stats -> iteration+1);
            //util::MemsetCASKernel<<<256, 256, 0, stream>>>(
            //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE),
            //    -2, -1,
            //    work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
            //        frontier_attribute->queue_index));
        }

        if (TO_TRACK)
        {
            util::Check_Value<<<1,1,0,stream>>>(
                work_progress -> template GetQueueLengthPointer<unsigned int>(
                    frontier_attribute->queue_index),
                data_slice->gpu_idx, 4, enactor_stats -> iteration);
            util::Check_Exist_<<<256, 256, 0, stream>>>(
                work_progress -> template GetQueueLengthPointer<unsigned int>(
                    frontier_attribute->queue_index),
                data_slice->gpu_idx, 4, enactor_stats -> iteration,
                frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));
            //util::Verify_Value_<<<256, 256, 0, stream>>>(
            //    data_slice -> gpu_idx, 4,
            //    work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
            //        frontier_attribute -> queue_index),
            //    enactor_stats -> iteration,
            //    frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE),
            //    data_slice -> labels.GetPointer(util::DEVICE),
            //    (Value)enactor_stats -> iteration+1);
        }
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
        if (false) //(Problem::ENABLE_IDEMPOTENCE && !Problem::MARK_PREDECESSORS && h_data_slice -> num_gpus > 1)
        {
            out_length[peer_] = num_elements;
            //printf("%d\t %lld\t %d\t incoming length = %d\n",
            //    h_data_slice -> gpu_idx, enactor_stats -> iteration, peer_,
            //    num_elements);
            return;
        }

        //printf("Expand_Incoming, num_elements = %d, queue_size = %d, size_check = %s, labels = %p\n",
        //    num_elements, keys_out.GetSize(), enactor -> size_check ? "true" : "false", h_data_slice -> labels.GetPointer(util::DEVICE));
        //fflush(stdout);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("ReceivedQueue",
        //    keys_in.GetPointer(util::DEVICE), num_elements,
        //    h_data_slice -> gpu_idx, iteration, peer_, stream);


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
            //VertexId iteration_ = iteration%2;
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
        Expand_Incoming_Kernel
            <AdvanceKernelPolicy, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
            (h_data_slice -> gpu_idx,
            iteration,
            num_elements,
            keys_in.GetPointer(util::DEVICE),
            vertex_associate_in.GetPointer(util::DEVICE),
            out_length.GetPointer(util::DEVICE) + ((enactor -> problem -> unified_receive) ? 0: peer_),
            keys_out.GetPointer(util::DEVICE),
            vertex_associate_orgs.GetPointer(util::DEVICE),
            h_data_slice -> labels.GetPointer(util::DEVICE),
            h_data_slice -> visited_mask.GetPointer(util::DEVICE));

        if (!enactor -> problem -> unified_receive)
            out_length.Move(util::DEVICE, util::HOST, 1, peer_, stream);
        else out_length.Move(util::DEVICE, util::HOST, 1, 0, stream);
        //if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
        //    "cudaStreamSynchronize failed", __FILE__, __LINE__))
        //    return;
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("ExpandedQueue",
        //    keys_out.GetPointer(util::DEVICE), out_length[peer_],
        //    h_data_slice -> gpu_idx, iteration, peer_, stream);
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
        //DataSlice                       data_slice,
        //DataSlice                      *d_data_slice,
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
        bool                            out_inv = false)
    {
        cudaError_t retval = cudaSuccess;
        //printf("SIZE_CHECK = %s\n", Enactor::SIZE_CHECK ? "true" : "false");
        bool over_sized = false;
        if ((!enactor -> size_check &&
            (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
            || (enactor -> problem -> data_slices[0] -> current_direction == BACKWARD))
            //(AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_FORWARD ||
            // AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_BACKWARD))
        {
            frontier_attribute -> output_length[0] = 0;
            return retval;

        } else {
            //printf("Size check runs\n");
            if (retval = Check_Size</*Enactor::SIZE_CHECK,*/ SizeT, SizeT> (
                enactor -> size_check,
                "scanned_edges", frontier_attribute->queue_length + 2,
                partitioned_scanned_edges, over_sized, -1, -1, -1, false))
                return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, Functor, gunrock::oprtr::advance::V2V>(
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
            return retval;
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
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (enactor -> debug)
        {
            printf("%d\t %d\t %d\t queue_length = %lld, output_length = %lld\n",
                thread_num, iteration, peer_,
                (long long)frontier_queue->keys[selector^1].GetSize(),
                (long long)request_length);
            fflush(stdout);
        }

        if (!enactor -> size_check &&
            (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
        {
            frontier_attribute -> output_length[0] = 0;
            return;
        } else if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
            //(AdvanceKernelPolicy::ADVANCE_MODE != gunrock::oprtr::advance::LB_CULL)
        {
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, VertexId > (
                    true, "queue3", request_length,
                    &frontier_queue->keys  [selector^1],
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, VertexId > (
                    true, "queue3", graph_slice->nodes+2,
                    &frontier_queue->keys  [selector  ],
                    over_sized, thread_num, iteration, peer_, true )) return;
            if (enactor -> problem -> use_double_buffer)
            {
                if (enactor_stats->retval =
                    Check_Size</*true,*/ SizeT, Value> (
                        true, "queue3", request_length,
                        &frontier_queue->values[selector^1],
                        over_sized, thread_num, iteration, peer_, false)) return;
                if (enactor_stats->retval =
                    Check_Size</*true,*/ SizeT, Value> (
                        true, "queue3", graph_slice->nodes+2,
                        &frontier_queue->values[selector  ],
                        over_sized, thread_num, iteration, peer_, true )) return;
            }
        } else {
            /*if (enactor_stats->retval =
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
            }*/
        }
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
        EnactorStats<SizeT>           *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        DataSlice* data_slice = data_slice_ -> GetPointer(util::HOST);
        if (false)//(Problem::ENABLE_IDEMPOTENCE && !Problem::MARK_PREDECESSORS && num_gpus > 1)
        {
            int num_blocks = (graph_slice -> nodes / 32 /  sizeof(MaskT)) / AdvanceKernelPolicy::THREADS + 1;
            if (num_blocks > 480) num_blocks = 480;
            Update_Mask_Kernel<Problem>
                <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>(
                graph_slice -> nodes,
                data_slice -> old_mask.GetPointer(util::DEVICE),
                data_slice -> visited_mask.GetPointer(util::DEVICE),
                data_slice -> labels.GetPointer(util::DEVICE));
            for (int gpu = 0; gpu < num_gpus; gpu++)
            {
                data_slice -> out_length[gpu] = (num_elements == 0) ?
                    0 : (graph_slice -> nodes / 8 / sizeof(MaskT)+1) * sizeof(MaskT) / sizeof(VertexId) + 1;
                data_slice -> in_length_out[gpu] = 0;
                data_slice -> keys_out[gpu].SetPointer(
                    (VertexId*)(data_slice -> visited_mask.GetPointer(util::DEVICE)),
                    (graph_slice -> nodes / (sizeof(MaskT)*8) + 1) * sizeof(MaskT) / sizeof(VertexId) + 1,
                    util::DEVICE);
                //printf("keys_out = %p\n",
                //    data_slice -> keys_out[gpu].GetPointer(util::DEVICE));
                data_slice -> keys_outs[gpu] = data_slice -> keys_out[gpu].GetPointer(util::DEVICE);
            }
        } else {
            BaseIteration::template Make_Output
                <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>(
                enactor,
                thread_num,
                num_elements,
                num_gpus,
                frontier_queue,
                scanned_edges,
                frontier_attribute,
                enactor_stats,
                data_slice_,
                graph_slice,
                work_progress,
                context,
                stream);
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

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam BfsEnactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
static CUT_THREADPROC BFSThread(
    void * thread_data_)
{
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value>
                                         GraphSlice;
    typedef BFSFunctor<VertexId, SizeT, Value, Problem>
                                         Functor   ;

    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data->problem;
    Enactor      *enactor            =  (Enactor*)     thread_data->enactor;
    int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
    DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    FrontierAttribute<SizeT>
                 *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    EnactorStats<SizeT>
                 *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

    if (enactor_stats[0].retval = util::SetDevice(gpu_idx))
    {
        thread_data -> status = ThreadSlice::Status::Ended;
        CUT_THREADEND;
    }

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

        for (int peer=0;peer<num_gpus;peer++)
        {
            frontier_attribute[peer].queue_index    = 0;        // Work queue index
            frontier_attribute[peer].queue_length   = peer==0?thread_data -> init_size:0;
            frontier_attribute[peer].selector       = 0; //frontier_attribute[peer].queue_length ==0 ? 0 : 1;
            frontier_attribute[peer].queue_reset    = true;
            enactor_stats     [peer].iteration      = 0;
        }

        gunrock::app::Iteration_Loop
            <Enactor, Functor,
            BFSIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>,
            Problem::MARK_PREDECESSORS ? 1 : 0, 0>
            (thread_data);
        // printf("BFS_Thread finished\n");fflush(stdout);
        thread_data -> status = ThreadSlice::Status::Idle;
    }

    thread_data->status = ThreadSlice::Status::Ended;
    CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 *
 * @tparam _Problem Problem type we process on.
 */
template <typename _Problem>
class BFSEnactor :
    public EnactorBase<typename _Problem::SizeT/*, _DEBUG, _SIZE_CHECK*/>
{
    ThreadSlice  *thread_slices;
    CUTThread    *thread_Ids   ;

public:
    _Problem     *problem      ;
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef typename Problem::MaskT    MaskT   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    typedef BFSEnactor<Problem>        Enactor;

    bool direction_optimized;
    float do_a, do_b;
    // Methods

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(
        int   num_gpus   = 1,
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true,
        bool  _direction_optimized = false) :
        BaseEnactor(
            VERTEX_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        thread_slices (NULL),
        thread_Ids    (NULL),
        problem       (NULL),
        direction_optimized (_direction_optimized),
        do_a          (0.001),
        do_b          (0.200)
    {
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (thread_slices != NULL)
        {
            for (int gpu = 0; gpu < this->num_gpus; gpu++)
                thread_slices[gpu].status = ThreadSlice::Status::ToKill;
            cutWaitForThreads(thread_Ids, this->num_gpus);
            delete[] thread_Ids   ; thread_Ids    = NULL;
            delete[] thread_slices; thread_slices = NULL;
        }
        if (retval = BaseEnactor::Release()) return retval;
        problem = NULL;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */

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
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitBFS(
        ContextPtr *context,
        Problem    *problem,
        int        max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = BaseEnactor::Init(
            //problem,
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].thread_num    = gpu;
            thread_slices[gpu].problem       = (void*)problem;
            thread_slices[gpu].enactor       = (void*)this;
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].status        = ThreadSlice::Status::Inited;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(BFSThread<
                    AdvanceKernelPolicy,FilterKernelPolicy,
                    BFSEnactor<Problem> >),
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
        if (retval =  BaseEnactor::Reset())
            return retval;
        for (int gpu=0; gpu < this->num_gpus; gpu++)
        {
            thread_slices[gpu].status = ThreadSlice::Status::Wait;
        }
        return retval;
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
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
    cudaError_t EnactBFS(
        VertexId    src)
    {
        clock_t      start_time = clock();
        cudaError_t  retval     = cudaSuccess;

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
                 thread_slices[gpu].init_size=1;
            else thread_slices[gpu].init_size=0;
            this->frontier_attribute[gpu*this->num_gpus].queue_length
                = thread_slices[gpu].init_size;
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
            }
        }

        for (int gpu=0; gpu<this->num_gpus * this -> num_gpus;gpu++)
        if (this->enactor_stats[gpu].retval!=cudaSuccess)
        {
            retval=this->enactor_stats[gpu].retval;
            return retval;
        }

        if (this -> debug) printf("\nGPU BFS Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        0,                                  // SATURATION QUIT
        true,                               // DEQUEUE_PROBLEM_SIZE
        (sizeof(VertexId) == 4) ? 8 : 4,    // MIN_CTA_OCCUPANCY
        8,                                  // LOG_THREADS
        2,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        5,                                  // END_BITMASK_CULL
        8,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::filter::CULL>
    FilterKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        8,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    TWCAdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    TWCAdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        8,                                  // MIN_CTA_OCCUPANCY
        8,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LBAdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
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
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    LB_LIGHT_AdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
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
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_CULL>
    LB_CULL_AdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
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
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT_CULL>
    LB_LIGHT_CULL_AdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT_CULL>
    LB_LIGHT_CULL_AdvanceKernelPolicy;

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename AdvanceKernelPolicy_IDEM,
        typename FilterKernelPolicy_IDEM,
        bool ENABLE_IDEMPOTENCE>
    struct IDEM_SWITCH{};

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy ,
        typename AdvanceKernelPolicy_IDEM,
        typename FilterKernelPolicy_IDEM>
    struct IDEM_SWITCH<
        AdvanceKernelPolicy, FilterKernelPolicy,
        AdvanceKernelPolicy_IDEM, FilterKernelPolicy_IDEM, true>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBFS<AdvanceKernelPolicy_IDEM, FilterKernelPolicy_IDEM>(src);
        }

        static cudaError_t Init(
            Enactor &enactor,
            ContextPtr  *context,
            Problem     *problem,
            int         max_grid_size  = 0)
        {
            return enactor.InitBFS<AdvanceKernelPolicy_IDEM, FilterKernelPolicy_IDEM>(context, problem, max_grid_size);
        }
    };

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy ,
        typename AdvanceKernelPolicy_IDEM,
        typename FilterKernelPolicy_IDEM>
    struct IDEM_SWITCH<
        AdvanceKernelPolicy, FilterKernelPolicy,
        AdvanceKernelPolicy_IDEM, FilterKernelPolicy_IDEM, false>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }

        static cudaError_t Init(
            Enactor &enactor,
            ContextPtr  *context,
            Problem     *problem,
            int         max_grid_size  = 0)
        {
            return enactor.InitBFS<AdvanceKernelPolicy, FilterKernelPolicy>(context, problem, max_grid_size);
        }
    };

    template <
        bool ENABLE_IDEMPOTENCE,
        gunrock::oprtr::advance::MODE A_MODE>
    struct MODE_SWITCH{};

    template <bool ENABLE_IDEMPOTENCE>
    struct MODE_SWITCH<ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB>
    {
        typedef IDEM_SWITCH<
                LBAdvanceKernelPolicy     , FilterKernelPolicy,
                LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy,
                ENABLE_IDEMPOTENCE> I_SWITCH;

        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return I_SWITCH::Enact(enactor, src);
        }

        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return I_SWITCH::Init(enactor, context, problem, max_grid_size);
        }
    };

    template <bool ENABLE_IDEMPOTENCE>
    struct MODE_SWITCH<ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::TWC_FORWARD>
    {
        typedef IDEM_SWITCH<
                TWCAdvanceKernelPolicy     , FilterKernelPolicy,
                TWCAdvanceKernelPolicy_IDEM, FilterKernelPolicy,
                ENABLE_IDEMPOTENCE> I_SWITCH;

        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return I_SWITCH::Enact(enactor, src);
        }

        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return I_SWITCH::Init(enactor, context, problem, max_grid_size);
        }
    };

    template <bool ENABLE_IDEMPOTENCE>
    struct MODE_SWITCH<ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_LIGHT>
    {
        typedef IDEM_SWITCH<
                LB_LIGHT_AdvanceKernelPolicy     , FilterKernelPolicy,
                LB_LIGHT_AdvanceKernelPolicy_IDEM, FilterKernelPolicy,
                ENABLE_IDEMPOTENCE> I_SWITCH;

        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return I_SWITCH::Enact(enactor, src);
        }

        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return I_SWITCH::Init(enactor, context, problem, max_grid_size);
        }
    };

    template <bool ENABLE_IDEMPOTENCE>
    struct MODE_SWITCH<ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_CULL>
    {
        typedef IDEM_SWITCH<
                LB_CULL_AdvanceKernelPolicy     , FilterKernelPolicy,
                LB_CULL_AdvanceKernelPolicy_IDEM, FilterKernelPolicy,
                ENABLE_IDEMPOTENCE> I_SWITCH;

        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return I_SWITCH::Enact(enactor, src);
        }

        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return I_SWITCH::Init(enactor, context, problem, max_grid_size);
        }
    };

    template <bool ENABLE_IDEMPOTENCE>
    struct MODE_SWITCH<ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_LIGHT_CULL>
    {
        typedef IDEM_SWITCH<
                LB_LIGHT_CULL_AdvanceKernelPolicy     , FilterKernelPolicy,
                LB_LIGHT_CULL_AdvanceKernelPolicy_IDEM, FilterKernelPolicy,
                ENABLE_IDEMPOTENCE> I_SWITCH;

        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return I_SWITCH::Enact(enactor, src);
        }

        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return I_SWITCH::Init(enactor, context, problem, max_grid_size);
        }
    };

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @param[in] src Source node to start primitive.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Enact(
        VertexId    src,
        std::string traversal_mode = "LB")
    {
        if (this -> min_sm_version >= 300)
        {
            if (this -> min_sm_version < 350 && sizeof(VertexId) == 8)
                return util::GRError("64bit-VertexId disabled, because atomicMin(long long) is only supported by compute capability 3.5 or higher",
                    __FILE__, __LINE__);
            else if (traversal_mode == "LB")
                return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB>
                    ::Enact(*this, src);
            else if (traversal_mode == "TWC")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::TWC_FORWARD>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_CULL>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_LIGHT>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Enact(*this, src);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }

    /**
     * @brief BFS Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size  = 0,
        std::string traversal_mode = "LB")
    {
        if (this -> min_sm_version >= 300)
        {
            if (this -> min_sm_version < 350 && sizeof(VertexId) == 8)
                return util::GRError("64bit-VertexId disabled, because atomicMin(long long) is only supported by compute capability 3.5 or higher",
                    __FILE__, __LINE__);
            else if (traversal_mode == "LB")
                return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "TWC")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::TWC_FORWARD>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_LIGHT>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<Problem::ENABLE_IDEMPOTENCE, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Init(*this, context, problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }

    /** @} */
};

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
