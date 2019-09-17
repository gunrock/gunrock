// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_kernel.cuh
 *
 * @brief kernel functions for base graph problem enactor
 */

#pragma once

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

#include <gunrock/util/scan/block_scan.cuh>

namespace gunrock {
namespace app {

/*
 * @brief Copy predecessor function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] keys Pointer to the key array.
 * @param[in] in_preds Pointer to the input predecessor array.
 * @param[out] out_preds Pointer to the output predecessor array.
 */
template <typename VertexT, typename SizeT>
__global__ void CopyPreds_Kernel(const SizeT num_elements, const VertexT* keys,
                                 const VertexT* in_preds, VertexT* out_preds) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

  while (x < num_elements) {
    VertexT t = keys[x];
    out_preds[t] = in_preds[t];
    x += STRIDE;
  }
}

/*
 * @brief Update predecessor function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] nodes Number of nodes in graph.
 * @param[in] keys Pointer to the key array.
 * @param[in] org_vertexs
 * @param[in] in_preds Pointer to the input predecessor array.
 * @param[out] out_preds Pointer to the output predecessor array.
 */
template <typename VertexId, typename SizeT>
__global__ void UpdatePreds_Kernel(const SizeT num_elements, const SizeT nodes,
                                   const VertexId* keys,
                                   const VertexId* org_vertexs,
                                   const VertexId* in_preds,
                                   VertexId* out_preds) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  VertexId x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  VertexId t, p;

  while (x < num_elements) {
    t = keys[x];
    p = in_preds[t];
    if (p < nodes) out_preds[t] = org_vertexs[p];
    x += STRIDE;
  }
}

/*
 * @brief Assign marker function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] num_gpus Number of GPUs used for testing.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] partition_table Pointer to the partition table.
 * @param[out] marker
 */
/*template <typename VertexId, class SizeT>
__global__ void Assign_Marker(
    const SizeT            num_elements,
    const int              num_gpus,
    const VertexId* const  keys_in,
    const int*      const  partition_table,
          SizeT**          marker)
{
    VertexId key;
    int gpu;
    //extern __shared__ SizeT* s_marker[];
    SharedMemory<SizeT*> smem;
    SizeT** s_marker = smem.getPointer();
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x= (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_gpus)
        s_marker[threadIdx.x] = marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        gpu = partition_table[key];
        for (int i=0;i<num_gpus;i++)
            s_marker[i][x]=(i==gpu)?1:0;
        x += STRIDE;
    }
}*/

/*
 * @brief Assign marker backward function.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_elements Number of elements in the array.
 * @param[in] num_gpus Number of GPUs used for testing.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] offsets Pointer to
 * @param[in] partition_table Pointer to the partition table.
 * @param[out] marker
 */
/*template <typename VertexId, class SizeT>
__global__ void Assign_Marker_Backward(
    const SizeT            num_elements,
    const int              num_gpus,
    const VertexId* const  keys_in,
    const SizeT*    const  offsets,
    const int*      const  partition_table,
          SizeT**          marker)
{
    VertexId key;
    //extern __shared__ SizeT* s_marker[];
    SharedMemory<SizeT*> smem;
    SizeT** s_marker = smem.getPointer();
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT x= (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_gpus)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        for (int gpu=0; gpu<num_gpus; gpu++)
            s_marker[gpu][x]=0;
        if (key!=-1) for (SizeT i=offsets[key];i<offsets[key+1];i++)
            s_marker[partition_table[i]][x]=1;
        x+=STRIDE;
    }
}*/

/*
 * @brief Make output function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam num_vertex_associates
 * @tparam num_value__associates
 *
 * @param[in] num_elements Number of elements.
 * @param[in] num_gpus Number of GPUs used.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] partition_table
 * @param[in] convertion_table
 * @param[in] array_size
 * @param[in] array
 */
template <typename VertexId, typename SizeT, typename Value,
          SizeT NUM_VERTEX_ASSOCIATES, SizeT NUM_VALUE__ASSOCIATES>
__global__ void Make_Out(const SizeT num_elements, const int num_gpus,
                         const VertexId* const keys_in,
                         const int* const partition_table,
                         const VertexId* const convertion_table,
                         const size_t array_size, char* array) {
  extern __shared__ char s_array[];
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  size_t offset = 0;
  SizeT** s_marker = (SizeT**)&(s_array[offset]);
  offset += sizeof(SizeT*) * num_gpus;
  VertexId** s_keys_outs = (VertexId**)&(s_array[offset]);
  offset += sizeof(VertexId*) * num_gpus;
  VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
  offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
  Value** s_value__associate_orgs = (Value**)&(s_array[offset]);
  offset += sizeof(Value*) * NUM_VALUE__ASSOCIATES;
  VertexId** s_vertex_associate_outss = (VertexId**)&(s_array[offset]);
  offset += sizeof(VertexId*) * num_gpus * NUM_VERTEX_ASSOCIATES;
  Value** s_value__associate_outss = (Value**)&(s_array[offset]);
  offset += sizeof(Value*) * num_gpus * NUM_VALUE__ASSOCIATES;
  SizeT* s_offset = (SizeT*)&(s_array[offset]);
  SizeT x = threadIdx.x;

  while (x < array_size) {
    s_array[x] = array[x];
    x += blockDim.x;
  }
  __syncthreads();

  x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  while (x < num_elements) {
    VertexId key = keys_in[x];
    int target = partition_table[key];
    SizeT pos = s_marker[target][x] - 1 + s_offset[target];

    if (target == 0) {
      s_keys_outs[0][pos] = key;
    } else {
      s_keys_outs[target][pos] = convertion_table[key];
#pragma unroll
      for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++)
        s_vertex_associate_outss[target * NUM_VERTEX_ASSOCIATES + i][pos] =
            s_vertex_associate_orgs[i][key];
#pragma unroll
      for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++)
        s_value__associate_outss[target * NUM_VALUE__ASSOCIATES + i][pos] =
            s_value__associate_orgs[i][key];
    }
    x += STRIDE;
  }
}

/*
 * @brief Make output backward function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam num_vertex_associates
 * @tparam num_value__associates
 *
 * @param[in] num_elements Number of elements.
 * @param[in] num_gpus Number of GPUs used.
 * @param[in] keys_in Pointer to the key array.
 * @param[in] partition_table
 * @param[in] convertion_table
 * @param[in] array_size
 * @param[in] array
 */
template <typename VertexId, typename SizeT, typename Value,
          SizeT NUM_VERTEX_ASSOCIATES, SizeT NUM_VALUE__ASSOCIATES>
__global__ void Make_Out_Backward(const SizeT num_elements, const int num_gpus,
                                  const VertexId* const keys_in,
                                  const SizeT* const offsets,
                                  const int* const partition_table,
                                  const VertexId* const convertion_table,
                                  const size_t array_size, char* array) {
  extern __shared__ char s_array[];
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  size_t offset = 0;
  SizeT** s_marker = (SizeT**)&(s_array[offset]);
  offset += sizeof(SizeT*) * num_gpus;
  VertexId** s_keys_outs = (VertexId**)&(s_array[offset]);
  offset += sizeof(VertexId*) * num_gpus;
  VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
  offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
  Value** s_value__associate_orgs = (Value**)&(s_array[offset]);
  offset += sizeof(Value*) * NUM_VALUE__ASSOCIATES;
  VertexId** s_vertex_associate_outss = (VertexId**)&(s_array[offset]);
  offset += sizeof(VertexId*) * num_gpus * NUM_VERTEX_ASSOCIATES;
  Value** s_value__associate_outss = (Value**)&(s_array[offset]);
  offset += sizeof(Value*) * num_gpus * NUM_VALUE__ASSOCIATES;
  SizeT* s_offset = (SizeT*)&(s_array[offset]);
  SizeT x = threadIdx.x;

  while (x < array_size) {
    s_array[x] = array[x];
    x += blockDim.x;
  }
  __syncthreads();

  x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  while (x < num_elements) {
    VertexId key = keys_in[x];
    if (key < 0) {
      x += STRIDE;
      continue;
    }
    for (SizeT j = offsets[key]; j < offsets[key + 1]; j++) {
      int target = partition_table[j];
      SizeT pos = s_marker[target][x] - 1 + s_offset[target];

      if (target == 0) {
        s_keys_outs[0][pos] = key;
      } else {
        s_keys_outs[target][pos] = convertion_table[j];
#pragma unroll
        for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++)
          s_vertex_associate_outss[target * NUM_VERTEX_ASSOCIATES + i][pos] =
              s_vertex_associate_orgs[i][key];
#pragma unroll
        for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++)
          s_value__associate_outss[target * NUM_VALUE__ASSOCIATES + i][pos] =
              s_value__associate_orgs[i][key];
      }
    }
    x += STRIDE;
  }
}

template <typename VertexT, typename SizeT, typename ValueT,
          SizeT NUM_VERTEX_ASSOCIATES, SizeT NUM_VALUE__ASSOCIATES>
__global__ void MakeOutput_Kernel(
    SizeT num_elements, int num_gpus, SizeT* d_out_length, VertexT* d_keys_in,
    int* d_partition_table, VertexT* d_convertion_table,
    VertexT** d_vertex_associate_orgs, ValueT** d_value__associate_orgs,
    VertexT** d_keys_outs, VertexT** d_vertex_associate_outs,
    ValueT** d_value__associate_outs, bool skip_convertion = false) {
  typedef util::Block_Scan<SizeT, 9> BlockScanT;
  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT sum_offset[8];
  //__shared__ SizeT offset[8];
  //__shared__ SizeT offset_zero;

  SizeT in_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (in_pos - threadIdx.x < num_elements) {
    VertexT key = util::PreDefinedValues<VertexT>::InvalidValue;
    int target = util::PreDefinedValues<int>::InvalidValue;
    if (in_pos < num_elements) {
      key = d_keys_in[in_pos];
      target = d_partition_table[key];
    }
    SizeT out_pos = 0, out_offset = 0;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      // SizeT out_offset;
      //__syncthreads();
      BlockScanT::LogicScan((target == gpu) ? 1 : 0, out_offset, scan_space);
      if (target == gpu) out_pos = out_offset;
      if (threadIdx.x == blockDim.x - 1) {
        // sum[gpu] = block_sum;
        sum_offset[gpu] = out_offset + ((target == gpu) ? 1 : 0);
        // offset[gpu] = atomicAdd(d_out_length + gpu, block_sum);
        // printf("sum[%d] -> %d\n", gpu, sum[gpu]);
        // if (gpu == 0) offset_zero = offset[gpu];
      }
      __syncthreads();
    }
    //__syncthreads();
    if (threadIdx.x < num_gpus) {
      // printf("sum[%d] = %d\n", threadIdx.x, sum[threadIdx.x]);
      sum_offset[threadIdx.x] =
          atomicAdd(d_out_length + threadIdx.x, sum_offset[threadIdx.x]);
    }
    __syncthreads();

    if (in_pos >= num_elements) break;
    // printf("(%4d, %4d) : in_pos = %d, key = %d, target = %d, out_pos = %d +
    // %d\n",
    //    blockIdx.x, threadIdx.x, in_pos, key, target, out_pos,
    //    offset[target]-1);
    if (!util::isValid(key)) {
      in_pos += STRIDE;
      continue;
    }
    out_pos += sum_offset[target] - 1;
    if (skip_convertion) {
      d_keys_outs[target][out_pos] = key;
    } else {
      d_keys_outs[target][out_pos] = d_convertion_table[key];
    }

    if (target != 0) {
      out_offset = out_pos * NUM_VERTEX_ASSOCIATES;
      //#pragma unroll
      for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++) {
        d_vertex_associate_outs[target][out_offset] =
            d_vertex_associate_orgs[i][key];
        out_offset++;
      }
      out_offset = out_pos * NUM_VALUE__ASSOCIATES;
      //#pragma unroll
      for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++) {
        d_value__associate_outs[target][out_offset] =
            d_value__associate_orgs[i][key];
        out_offset++;
      }
    }
    in_pos += STRIDE;
  }
}

template <typename VertexT, typename SizeT, typename ValueT,
          SizeT NUM_VERTEX_ASSOCIATES, SizeT NUM_VALUE__ASSOCIATES>
__global__ void MakeOutput_Backward_Kernel(
    SizeT num_elements, int num_gpus, SizeT* d_out_length, VertexT* d_keys_in,
    SizeT* d_offsets, int* d_partition_table, VertexT* d_convertion_table,
    VertexT** d_vertex_associate_orgs, ValueT** d_value__associate_orgs,
    VertexT** d_keys_outs, VertexT** d_vertex_associate_outs,
    ValueT** d_value__associate_outs, bool skip_convertion = false) {
  typedef util::Block_Scan<SizeT, 9> BlockScanT;
  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT sum_offset[8];
  SizeT out_pos[8];
  unsigned char gpu_select[8];
  //__shared__ SizeT offset[8];
  //__shared__ SizeT offset_zero;

  SizeT in_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (in_pos - threadIdx.x < num_elements) {
    VertexT key = util::PreDefinedValues<VertexT>::InvalidValue;
    for (int gpu = 0; gpu < num_gpus; gpu++) gpu_select[gpu] = 0;
    if (in_pos < num_elements) {
      key = d_keys_in[in_pos];
      for (int i = d_offsets[key]; i < d_offsets[key + 1]; i++)
        gpu_select[d_partition_table[i]] = 1;
    }
    // SizeT    out_pos = 0, out_offset = 0;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      BlockScanT::LogicScan(gpu_select[gpu], out_pos[gpu], scan_space);
      if (threadIdx.x == blockDim.x - 1) {
        sum_offset[gpu] = out_pos[gpu] + gpu_select[gpu];
      }
      __syncthreads();
    }
    if (threadIdx.x < num_gpus) {
      if (sum_offset[threadIdx.x] != 0)
        sum_offset[threadIdx.x] =
            atomicAdd(d_out_length + threadIdx.x, sum_offset[threadIdx.x]);
    }
    __syncthreads();

    if (in_pos >= num_elements) break;
    if (!util::isValid(key)) {
      in_pos += STRIDE;
      continue;
    }

    for (int i = d_offsets[key]; i < d_offsets[key + 1]; i++) {
      int target = d_partition_table[i];
      out_pos[target] += sum_offset[target] - 1;
      if (skip_convertion) {
        d_keys_outs[target][out_pos[target]] = key;
      } else {
        d_keys_outs[target][out_pos[target]] = d_convertion_table[i];
      }

      if (target != 0) {
        SizeT out_offset = out_pos[target] * NUM_VERTEX_ASSOCIATES;
        //#pragma unroll
        for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++) {
          d_vertex_associate_outs[target][out_offset] =
              d_vertex_associate_orgs[i][key];
          out_offset++;
        }
        out_offset = out_pos[target] * NUM_VALUE__ASSOCIATES;
        //#pragma unroll
        for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++) {
          d_value__associate_outs[target][out_offset] =
              d_value__associate_orgs[i][key];
          out_offset++;
        }
        // printf("Make_Output : values[%2d, %2d] = %.4f, %.4f\n",
        //    key, out_pos[target], d_value__associate_orgs[0][key],
        //    d_value__associate_orgs[1][key]);
      }
    }
    in_pos += STRIDE;
  }
}

template <typename VertexT, typename SizeT, typename ValueT,
          SizeT NUM_VERTEX_ASSOCIATES, SizeT NUM_VALUE__ASSOCIATES>
__global__ void MakeOutput_SkipSelection_Kernel(
    SizeT num_elements,
    // int        num_gpus,
    VertexT* d_keys_in, VertexT** d_vertex_associate_orgs,
    ValueT** d_value__associate_orgs, VertexT* d_keys_out,
    VertexT* d_vertex_associate_out, ValueT* d_value__associate_out) {
  SizeT in_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (in_pos < num_elements) {
    VertexT key = d_keys_in[in_pos];
    SizeT out_pos = in_pos;

    // keys_out[0][out_pos] = key;
    d_keys_out[out_pos] = key;
    SizeT temp_out = out_pos * NUM_VERTEX_ASSOCIATES;
#pragma unroll
    for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++) {
      d_vertex_associate_out[temp_out] = d_vertex_associate_orgs[i][key];
      temp_out++;
    }
    temp_out = out_pos * NUM_VALUE__ASSOCIATES;
#pragma unroll
    for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++) {
      d_value__associate_out[temp_out] = d_value__associate_orgs[i][key];
      temp_out++;
    }
    in_pos += STRIDE;
  }
}

template <typename VertexT, typename SizeT, typename ValueT,
          int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES,
          typename ExpandOpT>
__global__ void ExpandIncoming_Kernel(int gpu_num, SizeT num_elements,
                                      VertexT* keys_in,
                                      VertexT* vertex_associate_in,
                                      ValueT* value__associate_in,
                                      SizeT* out_length, VertexT* keys_out,
                                      ExpandOpT expand_op) {
  typedef util::Block_Scan<SizeT, 9> BlockScanT;

  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;
  SizeT in_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (in_pos - threadIdx.x < num_elements) {
    bool to_process = true;
    SizeT out_pos = util::PreDefinedValues<SizeT>::InvalidValue;
    VertexT key = util::PreDefinedValues<VertexT>::InvalidValue;

    if (in_pos < num_elements) {
      key = keys_in[in_pos];
      to_process =
          expand_op(key, in_pos, vertex_associate_in, value__associate_in);
    } else
      to_process = false;

    BlockScanT::LogicScan(to_process, out_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      block_offset = atomicAdd(out_length, out_pos + ((to_process) ? 1 : 0));
    }
    __syncthreads();

    if (to_process && keys_out != NULL) {
      out_pos += block_offset;
      keys_out[out_pos] = key;
    }
    in_pos += STRIDE;
  }
}

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
