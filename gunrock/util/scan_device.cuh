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
#include <moderngpu/kernel_scan.hxx>
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

template <typename InputT, typename OutputT, typename SizeT>
cudaError_t cubExclusiveSum(util::Array1D<uint64_t, char> &cub_temp_space,
                            util::Array1D<SizeT, InputT> &d_in,
                            util::Array1D<SizeT, OutputT> &d_out,
                            SizeT num_items, cudaStream_t stream = 0,
                            bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;
  size_t request_bytes = 0;

  retval = cub::DeviceScan::ExclusiveSum(
      NULL, request_bytes, d_in.GetPointer(util::DEVICE),
      d_out.GetPointer(util::DEVICE), num_items, stream, debug_synchronous);

  if (retval) return retval;

  retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::DeviceScan::ExclusiveSum(
      cub_temp_space.GetPointer(util::DEVICE), request_bytes,
      d_in.GetPointer(util::DEVICE), d_out.GetPointer(util::DEVICE), num_items,
      stream, debug_synchronous);

  if (retval) return retval;

  return retval;
}


template <typename InputT, typename OutputT, 
          typename ReduceT, typename SizeT>
cudaError_t Scan(util::Array1D<SizeT, InputT> &d_in, SizeT num_items,
                 util::Array1D<SizeT, OutputT> &d_out,
                 ReduceT *r, mgpu::context_t *context,
                 bool debug_synchronous = false) {

  if (context == nullptr) {
      return cudaErrorInvalidValue;
  }
  cudaError_t retval = cudaSuccess; 

  // TODO: Experiment with these values and choose the best ones. We
  // could even choose to make them a parameter of util::Scan and choose
  // based on our usage
  typedef mgpu::launch_box_t<
      mgpu::arch_30_cta<256, 7>,
      mgpu::arch_35_cta<256, 7>,
      mgpu::arch_50_cta<256, 7>,
      mgpu::arch_60_cta<256, 7>,
      mgpu::arch_70_cta<256, 7>,
      mgpu::arch_75_cta<256, 7>
      > launch_t;
  mgpu::scan<mgpu::scan_type_inc, launch_t>(d_in.GetPointer(util::DEVICE), num_items,
                                  d_out.GetPointer(util::DEVICE),
                                  mgpu::plus_t<InputT>(), r,
                                  *context);

  if (debug_synchronous) GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

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
