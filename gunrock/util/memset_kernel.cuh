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


} // namespace util
} // namespace gunrock

