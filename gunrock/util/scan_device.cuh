// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * scan_utils.cuh
 *
 * @brief kenel utils used for a device scan.
 */

#pragma once
#include <cub/cub.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace util {

/**
 * \addtogroup PublicInterface
 * @{
 */

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

template <typename InputT, typename OutputT, typename SizeT>
cudaError_t cubInclusiveSum(util::Array1D<uint64_t, char> &cub_temp_space,
                            util::Array1D<SizeT, InputT> &d_in,
                            util::Array1D<SizeT, OutputT> &d_out,
                            SizeT num_items, cudaStream_t stream = 0,
                            bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;
  size_t request_bytes = 0;

  retval = cub::DeviceScan::InclusiveSum(
      NULL, request_bytes, d_in.GetPointer(util::DEVICE),
      d_out.GetPointer(util::DEVICE), num_items, stream, debug_synchronous);

  if (retval) return retval;

  retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::DeviceScan::InclusiveSum(
      cub_temp_space.GetPointer(util::DEVICE), request_bytes,
      d_in.GetPointer(util::DEVICE), d_out.GetPointer(util::DEVICE), num_items,
      stream, debug_synchronous);

  if (retval) return retval;

  return retval;
}

/** @} */

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
