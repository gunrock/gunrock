// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * markSegment.cuh
 *
 * @brief Simple markSegment Kernel
 */


/******************************************************************************
 * Simple markSegment Kernel
 ******************************************************************************/
#pragma once

namespace gunrock {
namespace util {


__global__ void markSegment(int *flag, int *vid, int length)
{
	const int STRIDE = gridDim.x * blockDim.x;
	for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1; idx < length-1; idx += STRIDE) {
    		flag[idx] = (vid[idx] != vid[idx-1]) ? 1 : 0;
  	} 
}


} // namespace util
} // namespace gunrock
