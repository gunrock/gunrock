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


/** @} */

} //util
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
