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

#pragma once

namespace gunrock {
namespace util {

/**
 * Memset a device vector.
 */
template <typename T>
__global__ void MemsetKernel(T *d_out, T value, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_out[idx] = value;
    }
}

/**
 * Memset a device vector using indices
 */
template <typename T>
__global__ void MemsetIdxKernel(T *d_out, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_out[idx] = idx;
    }
}

/**
 * Add to each element in the array. (only support type with operator +)
 */
template <typename T>
__global__ void MemsetAddKernel(T *d_out, T value, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_out[idx] += value;
    }
}

/**
 * Scale to each element in the array. (only support type with operator *)
 */
template <typename T>
__global__ void MemsetScaleKernel(T *d_out, T value, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_out[idx] *= value;
    }
}

/**
 * Add  source vector to destination vector. Two vectors should have the same length. (only support type with operator +)
 */
template <typename T>
__global__ void MemsetAddVectorKernel(T *d_dst, T *d_src, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_dst[idx] += d_src[idx];
    }
}

} // namespace util
} // namespace gunrock

