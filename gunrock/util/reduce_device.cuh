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
cudaError_t CUBSegReduce_sum(
    InputT 	*d_in,
    OutputT	*d_out,
    OffsetT	*d_offsets,
    SizeT 	num_segments)
{
    cudaError_t retval = cudaSuccess;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    if(util::GRError(
	    (retval = cub::DeviceSegmentedReduce::Sum(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_out,
		(int) num_segments,
		(int*)d_offsets,
		(int*)d_offsets+1)),
	    "CUBSegReduce cub::DeviceSegmentedReduce::Sum failed",
	    __FILE__, __LINE__)) return retval;
    // allocate temporary storage
    if (util::GRError(
            (retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
            "CUBSegReduce malloc d_temp_storage failed",
            __FILE__, __LINE__)) return retval;
    // run reduce
    if (util::GRError(
            (retval = cub::DeviceSegmentedReduce::Sum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                (int) num_segments,
		(int*)d_offsets,
		(int*)d_offsets+1)),
            "CUBSegReduce cub::DeviceSegmentedReduce::Sum failed",
            __FILE__, __LINE__)) return retval;

    // clean up
/*    if (util::GRError(
            (retval = cudaFree(d_temp_storage)),
            "CUBSegReduce free d_temp_storage failed",
            __FILE__, __LINE__)) return retval;
*/
    return retval;
}

template <typename InputT, typename OutputT,
    typename SizeT, typename ReductionOp>
cudaError_t cubSegmentedReduce(
    util::Array1D<SizeT, char   > &cub_temp_space,
    util::Array1D<SizeT, InputT > &keys_in,
    util::Array1D<SizeT, OutputT> &keys_out,
                         SizeT     num_segments,
    util::Array1D<SizeT, SizeT  > &segment_offsets,
                     ReductionOp   reduction_op,
                         InputT    initial_value,
                    cudaStream_t   stream = 0,
                         bool      debug_synchronous = false)
{
    cudaError_t retval = cudaSuccess;
    size_t request_bytes = 0;
    
    retval = cub::DispatchSegmentedReduce<InputT*, OutputT*, SizeT*,
        SizeT, ReductionOp>::Dispatch(
        NULL, request_bytes,
        keys_in.GetPointer(util::DEVICE),
        keys_out.GetPointer(util::DEVICE),
        num_segments,
        segment_offsets.GetPointer(util::DEVICE),
        segment_offsets.GetPointer(util::DEVICE) + 1,
        reduction_op, initial_value, stream, debug_synchronous);
    if (retval)
        return retval;

    retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
    if (retval)
        return retval;

    retval = cub::DispatchSegmentedReduce<InputT*, OutputT*, SizeT*,
        SizeT, ReductionOp>::Dispatch(
        cub_temp_space.GetPointer(util::DEVICE), request_bytes,
        keys_in.GetPointer(util::DEVICE),
        keys_out.GetPointer(util::DEVICE),
        num_segments,
        segment_offsets.GetPointer(util::DEVICE),
        segment_offsets.GetPointer(util::DEVICE) + 1,
        reduction_op, initial_value, stream, debug_synchronous);
    if (retval)
        return retval;
 
    return retval;
}

template <typename InputT, typename OutputT,
    typename SizeT, typename ReductionOp>
cudaError_t cubReduce(
    util::Array1D<SizeT, char   > &cub_temp_space,
    util::Array1D<SizeT, InputT > &keys_in,
    util::Array1D<SizeT, OutputT> &keys_out,
                         SizeT     num_keys,
                     ReductionOp   reduction_op,
                         InputT    initial_value,
                    cudaStream_t   stream = 0,
                         bool      debug_synchronous = false)
{
    cudaError_t retval = cudaSuccess;
    size_t request_bytes = 0;
    
    retval = cub::DispatchReduce<InputT*, OutputT*,
        SizeT, ReductionOp>::Dispatch(
        NULL, request_bytes,
        keys_in.GetPointer(util::DEVICE),
        keys_out.GetPointer(util::DEVICE),
        num_keys,
        reduction_op, initial_value, stream, debug_synchronous);
    if (retval)
        return retval;

    retval = cub_temp_space.EnsureSize_(request_bytes, util::DEVICE);
    if (retval)
        return retval;

    retval = cub::DispatchReduce<InputT*, OutputT*,
        SizeT, ReductionOp>::Dispatch(
        cub_temp_space.GetPointer(util::DEVICE), request_bytes,
        keys_in.GetPointer(util::DEVICE),
        keys_out.GetPointer(util::DEVICE),
        num_keys,
        reduction_op, initial_value, stream, debug_synchronous);
    if (retval)
        return retval;
 
    return retval;
}

/** @} */

} //util
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
