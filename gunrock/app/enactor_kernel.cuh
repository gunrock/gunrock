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
template <typename VertexId, typename SizeT>
__global__ void Copy_Preds (
    const SizeT     num_elements,
    const VertexId* keys,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    VertexId x = (SizeT)blockIdx.x*blockDim.x+threadIdx.x;
    VertexId t;

    while (x<num_elements)
    {
        t = keys[x];
        out_preds[t] = in_preds[t];
        x+= STRIDE;
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
__global__ void Update_Preds (
    const SizeT     num_elements,
    const SizeT     nodes,
    const VertexId* keys,
    const VertexId* org_vertexs,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    VertexId x = (SizeT)blockIdx.x*blockDim.x + threadIdx.x;
    VertexId t, p;

    while (x<num_elements)
    {
        t = keys[x];
        p = in_preds[t];
        if (p<nodes) out_preds[t] = org_vertexs[p];
        x+= STRIDE;
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
template <typename VertexId, class SizeT>
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
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();

    while (x < num_elements)
    {
        key = keys_in[x];
        gpu = partition_table[key];
        for (int i=0;i<num_gpus;i++)
            s_marker[i][x]=(i==gpu)?1:0;
        x+=STRIDE;
    }
}

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
template <typename VertexId, class SizeT>
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
}

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
__global__ void Make_Out(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   ) * num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * num_gpus * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   ) * num_gpus * NUM_VALUE__ASSOCIATES;
    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        int      target = partition_table[key];
        SizeT    pos    = s_marker[target][x]-1 + s_offset[target];

        if (target==0)
        {
            s_keys_outs[0][pos]=key;
        } else {
            s_keys_outs[target][pos]=convertion_table[key];
            #pragma unroll
            for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++)
                s_vertex_associate_outss[ target * NUM_VERTEX_ASSOCIATES + i][pos]
                    =s_vertex_associate_orgs[i][key];
            #pragma unroll
            for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++)
                s_value__associate_outss[ target * NUM_VALUE__ASSOCIATES + i][pos]
                    =s_value__associate_orgs[i][key];
        }
        x+=STRIDE;
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
__global__ void Make_Out_Backward(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  SizeT*      const offsets,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   ) * num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * num_gpus * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   ) * num_gpus * NUM_VALUE__ASSOCIATES;
    SizeT*     s_offset                = (SizeT*    )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        if (key < 0)
        {
            x+=STRIDE;
            continue;
        }
        for (SizeT j = offsets[key]; j < offsets[key+1]; j++)
        {
            int      target = partition_table[j];
            SizeT    pos    = s_marker[target][x]-1 + s_offset[target];

            if (target==0)
            {
                s_keys_outs[0][pos]=key;
            } else {
                s_keys_outs[target][pos]=convertion_table[j];
                #pragma unroll
                for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++)
                    s_vertex_associate_outss[ target * NUM_VERTEX_ASSOCIATES + i][pos]
                        =s_vertex_associate_orgs[i][key];
                #pragma unroll
                for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++)
                    s_value__associate_outss[ target * NUM_VALUE__ASSOCIATES + i][pos]
                        =s_value__associate_orgs[i][key];
            }
        }
        x+=STRIDE;
    }
}

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
