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
template <typename T>
__global__ void MemsetKernel(T *d_out, T value, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_out[idx] = value;
    }
}

/**
 * @brief Memset a device vector with the element's index in the vector
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] length Vector length
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
 * @brief Add value to each element in a device vector.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Value we want to add to each element in the vector
 * @param[in] length Vector length
 */
template <typename T>
__global__ void MemsetAddKernel(T *d_out, T value, int length)
{
    const int STRIDE = gridDim.x * blockDim.x; for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
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
template <typename T>
__global__ void MemsetScaleKernel(T *d_out, T value, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
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
template <typename T>
__global__ void MemsetAddVectorKernel(T *d_dst, T *d_src, int length)
{
    const int STRIDE = gridDim.x * blockDim.x;
    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
        d_dst[idx] += d_src[idx];
    }
}

/** @} */

} // namespace util
} // namespace gunrock

