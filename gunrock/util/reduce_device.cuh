// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * reduce_utils.cuh
 *
 * @brief kenel utils used in subgraph matching, triangle counting algorithms.
 */

#pragma once
#include <cub/cub.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <gunrock/util/reduction/kernel.cuh>

namespace gunrock {
namespace util {

/**
 * \addtogroup PublicInterface
 * @{
 */

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

template <typename InputT, typename OutputT, typename SizeT, typename OffsetT>
cudaError_t CUBSegReduce_sum(InputT *d_in, OutputT *d_out, OffsetT *d_offsets,
                             SizeT num_segments) {
  cudaError_t retval = cudaSuccess;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  if (util::GRError(
          (retval = cub::DeviceSegmentedReduce::Sum(
               d_temp_storage, temp_storage_bytes, d_in, d_out,
               (int)num_segments, (int *)d_offsets, (int *)d_offsets + 1)),
          "CUBSegReduce cub::DeviceSegmentedReduce::Sum failed", __FILE__,
          __LINE__))
    return retval;
  // allocate temporary storage
  if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
                    "CUBSegReduce malloc d_temp_storage failed", __FILE__,
                    __LINE__))
    return retval;
  // run reduce
  if (util::GRError(
          (retval = cub::DeviceSegmentedReduce::Sum(
               d_temp_storage, temp_storage_bytes, d_in, d_out,
               (int)num_segments, (int *)d_offsets, (int *)d_offsets + 1)),
          "CUBSegReduce cub::DeviceSegmentedReduce::Sum failed", __FILE__,
          __LINE__))
    return retval;

  // clean up
  /*    if (util::GRError(
              (retval = cudaFree(d_temp_storage)),
              "CUBSegReduce free d_temp_storage failed",
              __FILE__, __LINE__)) return retval;
  */
  return retval;
}

template <typename InputT, typename OutputT, typename SizeT,
          typename ReductionOp>
cudaError_t cubSegmentedReduce(util::Array1D<uint64_t, char> &cub_temp_space,
                               util::Array1D<SizeT, InputT> &keys_in,
                               util::Array1D<SizeT, OutputT> &keys_out,
                               SizeT num_segments,
                               util::Array1D<SizeT, SizeT> &segment_offsets,
                               ReductionOp reduction_op, InputT initial_value,
                               cudaStream_t stream = 0,
                               bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;
  size_t request_bytes = 0;

  retval = cub::DispatchSegmentedReduce<
      InputT *, OutputT *, SizeT *, SizeT,
      ReductionOp>::Dispatch(NULL, request_bytes,
                             keys_in.GetPointer(util::DEVICE),
                             keys_out.GetPointer(util::DEVICE), num_segments,
                             segment_offsets.GetPointer(util::DEVICE),
                             segment_offsets.GetPointer(util::DEVICE) + 1,
                             reduction_op, initial_value, stream,
                             debug_synchronous);
  if (retval) return retval;

  retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::DispatchSegmentedReduce<
      InputT *, OutputT *, SizeT *, SizeT,
      ReductionOp>::Dispatch(cub_temp_space.GetPointer(util::DEVICE),
                             request_bytes, keys_in.GetPointer(util::DEVICE),
                             keys_out.GetPointer(util::DEVICE), num_segments,
                             segment_offsets.GetPointer(util::DEVICE),
                             segment_offsets.GetPointer(util::DEVICE) + 1,
                             reduction_op, initial_value, stream,
                             debug_synchronous);
  if (retval) return retval;

  return retval;
}

template <typename InputT, typename OutputT, typename SizeT,
          typename ReductionOp>
cudaError_t cubReduce(util::Array1D<uint64_t, char> &cub_temp_space,
                      util::Array1D<SizeT, InputT> &keys_in,
                      util::Array1D<SizeT, OutputT> &keys_out, SizeT num_keys,
                      ReductionOp reduction_op, InputT initial_value,
                      cudaStream_t stream = 0, bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;
  size_t request_bytes = 0;

  retval =
      cub::DispatchReduce<InputT *, OutputT *, SizeT, ReductionOp>::Dispatch(
          NULL, request_bytes, keys_in.GetPointer(util::DEVICE),
          keys_out.GetPointer(util::DEVICE), num_keys, reduction_op,
          initial_value, stream, debug_synchronous);
  if (retval) return retval;

  retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval =
      cub::DispatchReduce<InputT *, OutputT *, SizeT, ReductionOp>::Dispatch(
          cub_temp_space.GetPointer(util::DEVICE), request_bytes,
          keys_in.GetPointer(util::DEVICE), keys_out.GetPointer(util::DEVICE),
          num_keys, reduction_op, initial_value, stream, debug_synchronous);
  if (retval) return retval;

  return retval;
}

template <typename InputT, typename OutputT, typename SizeT,
          typename ReductionOp>
cudaError_t SegmentedReduce(util::Array1D<uint64_t, char> &temp_space,
                            util::Array1D<SizeT, InputT> &keys_in,
                            util::Array1D<SizeT, OutputT> &keys_out,
                            SizeT num_segments,
                            util::Array1D<SizeT, SizeT> &segment_offsets,
                            ReductionOp reduction_op, OutputT initial_value,
                            cudaStream_t stream = 0,
                            bool debug_synchronous = false,
                            util::Location target = util::DEVICE) {
  cudaError_t retval = cudaSuccess;

  if ((target & util::HOST) != 0) {
#pragma omp parallel for
    for (SizeT seg = 0; seg < num_segments; seg++) {
      OutputT val = initial_value;
      SizeT seg_end = segment_offsets[seg + 1];
      for (SizeT pos = segment_offsets[seg]; pos < seg_end; pos++)
        val = reduction_op(val, keys_in[pos]);
      keys_out[seg] = val;
    }
  }

  if ((target & util::DEVICE) != 0) {
    uint64_t request_size = sizeof(SizeT) * (1 + num_segments);
    GUARD_CU(temp_space.EnsureSize_(request_size, util::DEVICE));
    SizeT *grid_segments = (SizeT *)(temp_space.GetPointer(util::DEVICE));
    int block_size = reduce::BLOCK_SIZE_;

    int grid_size = num_segments / block_size + 1;
    if (grid_size > 384) grid_size = 384;
    // util::PrintMsg("num_segments = " + std::to_string(num_segments)
    //    + ", request_size = " + std::to_string(request_size)
    //    + ", grid_size = " + std::to_string(grid_size)
    //    + ", block_size = " + std::to_string(reduce::BLOCK_SIZE));

    GUARD_CU(keys_in.ForAll(
        [grid_segments] __host__ __device__(InputT * keys, const SizeT &pos) {
          grid_segments[0] = 0;
        },
        (SizeT)1, util::DEVICE, stream));

    reduce::SegReduce_Kernel<<<grid_size, block_size, 0, stream>>>(
        keys_in.GetPointer(util::DEVICE), keys_out.GetPointer(util::DEVICE),
        num_segments, segment_offsets.GetPointer(util::DEVICE), reduction_op,
        initial_value, grid_segments, grid_segments + 1);

    // reduce::SegReduce_GInit
    //    <<< grid_size, reduce::BLOCK_SIZE, 0, stream >>> (
    //    grid_segments, grid_segments + 1,
    //    keys_out.GetPointer(util::DEVICE), initial_value);

    reduce::SegReduce_GKernel<<<grid_size, block_size, 0, stream>>>(
        keys_in.GetPointer(util::DEVICE), keys_out.GetPointer(util::DEVICE),
        num_segments, segment_offsets.GetPointer(util::DEVICE), reduction_op,
        initial_value, grid_segments, grid_segments + 1);
  }

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
