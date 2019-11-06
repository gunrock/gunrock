// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file: mark_segment.cuh
 *
 * @brief Simple mark_segmentation Kernel
 */

/******************************************************************************
 * Simple markSegment Kernel
 ******************************************************************************/
#pragma once

namespace gunrock {
namespace util {

/**
 * @brief flag array kernel with 1 indicating the start of
 * each segment and 0 otherwise
 *
 * @param[in] flag   Generated flag array
 * @param[in] vid    Input keys array used to generate flag array
 * @param[in] length Vector length
 */
template <typename T>
__global__ void MarkSegmentFromKeys(unsigned int *flag, T *vid, int len) {
  const int STRIDE = gridDim.x * blockDim.x;
  // skip the first one facilitate scan for keys array
  for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1; idx < len;
       idx += STRIDE) {
    flag[idx] = (vid[idx] != vid[idx - 1]) ? 1 : 0;
  }
}

/**
 * @brief flag array kernel with 1 indicating the start of
 * each segment and 0 otherwise
 *
 * @tparam T datatype of the input vector.
 *
 * @param[in] flag    Generated flag array
 * @param[in] offsets Input offsets used to generate flag array
 * @param[in] length  Vector length
 */
template <typename T>
__global__ void MarkSegmentFromIndices(unsigned int *flag, T *indices,
                                       int len) {
  const int STRIDE = gridDim.x * blockDim.x;
  // skip the first one facilitate scan for keys array
  for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1; idx < len;
       idx += STRIDE) {
    flag[indices[idx]] = 1;
  }
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variable
// mode:c++
// c-file-style: "NVIDIA"
// End:
