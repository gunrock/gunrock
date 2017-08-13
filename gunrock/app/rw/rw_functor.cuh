// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_functor.cuh
 *
 * @brief Device functions for rw problem.
 */

#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/rw/rw_problem.cuh>
#include <stdio.h>
#include <math.h>

namespace gunrock {
namespace app {
namespace rw {

/**
 * @brief Structure contains device functions in rw graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId >
struct RWFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;


    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. calculate output frontier in rw problem
     *
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
    }
};


/**
 * @brief Multiply the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename D, typename SizeT>
__global__ void RandomNext(T *paths, T *num_neighbor, D *d_rand, T *d_row_offsets, T *d_col_indices,
                                        SizeT length, SizeT itr)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        //printf("d_rand[%d]: %.6f -> %d\n", idx, d_rand[idx], temp);

        //this node : itr * length + idx,                   path[itr*length+idx]       -> node_id[idx]
        //result (next node) : (itr+1) * length +idx,       path[(itr+1)*length + idx] -> path[idx]


        //calculate offset in neighbor list
        SizeT node_id = paths[itr*length+idx];
        SizeT offset = __float2int_ru(num_neighbor[node_id] * d_rand[idx]) - 1;
        SizeT new_node = d_row_offsets[node_id] + offset;
        paths[(itr+1)*length + idx] = d_col_indices[new_node];



    }
};


/*
//combine two kernel
template <typename T, typename SizeT>
__global__ void MemsetAssignKernel(T *paths, T *d_row_offsets, T *d_col_indices, T *node_id, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        SizeT new_node = d_row_offsets[node_id[idx]] + paths[idx];
        paths[idx] = d_col_indices[new_node];
    }
};
*/


} // rw
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
