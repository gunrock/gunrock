// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * select_device.cuh
 *
 * @brief kenel utils used in minimum spanning tree, subgraph matching
 * algorithms.
 */

#pragma once
#include <cub/cub.cuh>

namespace gunrock {
namespace util {

/**
 * \addtogroup PublicInterface
 * @{
 */

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
struct GreaterThan {
  int compare;

  __host__ __device__ __forceinline__ GreaterThan(int compare)
      : compare(compare) {}

  __host__ __device__ __forceinline__ bool operator()(const int &a) const {
    return (a > compare);
  }
};

/**
 * @brief Uses the \p d_flags sequence to selectively copy the corresponding
 * items from \p d_in into \p d_out. The total number of items selected is
 * stored in \p d_num_selected_out.
 *
 */
template <typename InputT, typename OutputT, typename SizeT, typename Value,
          typename FlagT>
cudaError_t CUBSelect_flagged(InputT *d_in, FlagT *d_flags, OutputT *d_out,
                              Value *d_num_selected_out, SizeT num_elements) {
  cudaError_t retval = cudaSuccess;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  if (util::GRError((retval = cub::DeviceSelect::Flagged(
                         d_temp_storage, temp_storage_bytes, d_in, d_flags,
                         d_out, d_num_selected_out, num_elements)),
                    "CUBSelect_flagged cub::DeviceSelect::Flagged failed",
                    __FILE__, __LINE__))
    return retval;
  // allocate temporary storage
  if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
                    "CUBSelect malloc d_temp_storage failed", __FILE__,
                    __LINE__))
    return retval;
  // run selection
  if (util::GRError((retval = cub::DeviceSelect::Flagged(
                         d_temp_storage, temp_storage_bytes, d_in, d_flags,
                         d_out, d_num_selected_out, num_elements)),
                    "CUBSelect cub::DeviceSelect::Flagged failed", __FILE__,
                    __LINE__))
    return retval;

  // clean up
  /*    if (util::GRError(
              (retval = cudaFree(d_temp_storage)),
              "CUBSelect free d_temp_storage failed",
              __FILE__, __LINE__)) return retval;
  */
  return retval;
}

/**
 * @brief Uses the \p d_flags sequence to selectively copy the corresponding
 * items from \p d_in into \p d_out. The total number of items selected is
 * stored in \p d_num_selected_out.
 *
 */
template <typename InputT, typename OutputT, typename SizeT, typename SelectOp>
cudaError_t CUBSelect_if(InputT *d_in, OutputT *d_out,
                         SizeT *d_num_selected_out, SizeT num_elements) {
  GreaterThan select_op(0);
  cudaError_t retval = cudaSuccess;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  if (util::GRError((retval = cub::DeviceSelect::If(
                         d_temp_storage, temp_storage_bytes, d_in, d_out,
                         d_num_selected_out, num_elements, select_op)),
                    "CUBSelect_flagged cub::DeviceSelect::If failed", __FILE__,
                    __LINE__))
    return retval;
  // allocate temporary storage
  if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
                    "CUBSelect malloc d_temp_storage failed", __FILE__,
                    __LINE__))
    return retval;
  // run selection
  if (util::GRError((retval = cub::DeviceSelect::If(
                         d_temp_storage, temp_storage_bytes, d_in, d_out,
                         d_num_selected_out, num_elements, select_op)),
                    "CUBSelect cub::DeviceSelect::If failed", __FILE__,
                    __LINE__))
    return retval;

  // clean up
  /*    if (util::GRError(
              (retval = cudaFree(d_temp_storage)),
              "CUBSelect free d_temp_storage failed",
              __FILE__, __LINE__)) return retval;
  */
  return retval;
}

/**
 * @brief selects items from a sequence of int keys using a
 * section functor (greater-than)
 *
 */
template <typename T, typename SizeT>
cudaError_t CUBSelect(T *d_input, SizeT num_elements, T *d_output,
                      unsigned int *num_selected) {
  cudaError_t retval = cudaSuccess;
  SizeT *d_num_selected = NULL;

  if (util::GRError(
          (retval = cudaMalloc((void **)&d_num_selected, sizeof(SizeT))),
          "CUBSelect d_num_selected malloc failed", __FILE__, __LINE__))
    return retval;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  GreaterThan select_op(-1);

  // determine temporary device storage requirements
  if (util::GRError((retval = cub::DeviceSelect::If(
                         d_temp_storage, temp_storage_bytes, d_input, d_output,
                         d_num_selected, num_elements, select_op)),
                    "CUBSelect cub::DeviceSelect::If failed", __FILE__,
                    __LINE__))
    return retval;

  // allocate temporary storage
  if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
                    "CUBSelect malloc d_temp_storage failed", __FILE__,
                    __LINE__))
    return retval;

  // run selection
  if (util::GRError((retval = cub::DeviceSelect::If(
                         d_temp_storage, temp_storage_bytes, d_input, d_output,
                         d_num_selected, num_elements, select_op)),
                    "CUBSelect cub::DeviceSelect::If failed", __FILE__,
                    __LINE__))
    return retval;

  if (util::GRError(
          (retval = cudaMemcpy(num_selected, d_num_selected, sizeof(SizeT),
                               cudaMemcpyDeviceToHost)),
          "CUBSelect copy back num_selected failed", __FILE__, __LINE__))
    return retval;

  // clean up
  if (util::GRError((retval = cudaFree(d_temp_storage)),
                    "CUBSelect free d_temp_storage failed", __FILE__, __LINE__))
    return retval;
  if (util::GRError((retval = cudaFree(d_num_selected)),
                    "CUBSelect free d_num_selected failed", __FILE__, __LINE__))
    return retval;

  return retval;
}

template <typename InputT, typename OutputT, typename SizeT, typename SelectOp>
cudaError_t cubSelectIf(util::Array1D<uint64_t, char> &cub_temp_space,
                        util::Array1D<SizeT, InputT> &keys_in,
                        util::Array1D<SizeT, OutputT> &keys_out,
                        util::Array1D<SizeT, SizeT> &num_selected,
                        SizeT num_keys, SelectOp select_op,
                        cudaStream_t stream = 0,
                        bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;

  typedef cub::NullType *FlagIterator;
  typedef cub::NullType EqualityOp;

  size_t request_bytes = 0;
  retval = cub::DispatchSelectIf<
      InputT *, FlagIterator, OutputT *, SizeT *, SelectOp, EqualityOp, SizeT,
      false>::Dispatch(NULL, request_bytes, keys_in.GetPointer(util::DEVICE),
                       NULL, keys_out.GetPointer(util::DEVICE),
                       num_selected.GetPointer(util::DEVICE), select_op,
                       EqualityOp(), num_keys, stream, debug_synchronous);
  if (retval) return retval;

  retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::DispatchSelectIf<
      InputT *, FlagIterator, OutputT *, SizeT *, SelectOp, EqualityOp, SizeT,
      false>::Dispatch(cub_temp_space.GetPointer(util::DEVICE), request_bytes,
                       keys_in.GetPointer(util::DEVICE), NULL,
                       keys_out.GetPointer(util::DEVICE),
                       num_selected.GetPointer(util::DEVICE), select_op,
                       EqualityOp(), num_keys, stream, debug_synchronous);
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
