// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * memset_kernel.cuh
 *
 * @brief Simple Memset Kernel
 */


/******************************************************************************
 * Simple Memset Kernel
 ******************************************************************************/
#include <gunrock/coo.cuh>
#pragma once

namespace gunrock {
namespace util {

// TODO: The memset kernels are getting nasty.
// Need to use operator overload to rewrite most
// of these some day.

/**
 * \addtogroup PublicInterface
 * @{
 */

/**
 * @brief Memset a device vector.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Value we want to set
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetKernel(T *d_out, T value, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx] = value;
    }
}

template <typename VertexId, typename SizeT, typename Value>
__global__ void MemsetAddEdgeValKernel(Coo<VertexId, Value> *d_out, VertexId value, SizeT length)
{
   const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx].row += value;
        d_out[idx].col += value;
    } 
}

/**
 * @brief Memset a device vector with the element's index in the vector
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] length Vector length
 * @param[in] scale The scale for indexing (1 by default)
 */
template <typename T, typename SizeT>
__global__ void MemsetIdxKernel(T *d_out, SizeT length, T scale=1)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx] = idx * scale;
    }
}

/**
 * @brief Add value to each element in a device vector.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Value we want to add to each element in the vector
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetAddKernel(T *d_out, T value, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx] += value;
    }
}

/**
 * @brief Scale each element in a device vector to a certain factor.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Scale factor
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetScaleKernel(T *d_out, T value, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx] *= value;
    }
}

/**
 * @brief Add the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetAddVectorKernel(T *d_dst, T *d_src, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_dst[idx] += d_src[idx];
    }
}

/**
 * @brief Multiply the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetMultiplyVectorKernel(T *d_dst, T *d_src, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_dst[idx] *= d_src[idx];
    }
}

/**
 * @brief Divide the source vector to the destination vector with the same length
 * TODO: divide by zero check
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetDivVectorKernel(T *d_dst, T *d_src, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_dst[idx] /= d_src[idx];
    }
}

/**
 * @brief Copy the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void MemsetCopyVectorKernel(T *d_dst, T *d_src, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_dst[idx] = d_src[idx];
    }
}

/**
 * @brief Add the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src1 Source device-side vector 1
 * @param[in] d_src2 Source device-side vector 2
 * @param[in] scale Scale factor
 * @param[in] length Vector length
 */
template <typename T, typename SizeT>
__global__ void
MemsetMadVectorKernel(T *d_dst, T *d_src1, T *d_src2, T scale, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_dst[idx] = d_src1[idx] * scale + d_src2[idx];
    }
}

template <typename T, typename SizeT>
__global__ void MemsetCASKernel(T *d_dst, T compare, T val, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; 
        idx < length; idx += STRIDE) 
    {
        if (d_dst[idx] == compare) d_dst[idx] = val;
    }
}

template <typename T, typename SizeT>
__global__ void MemsetCASKernel(T *d_dst, T compare, T val, SizeT *length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; 
        idx < length[0]; idx += STRIDE) 
    {
        if (d_dst[idx] == compare) d_dst[idx] = val;
    }
}
/** @} */

} // namespace util
} // namespace gunrock
