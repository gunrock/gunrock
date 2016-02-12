// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * select_utils.cuh
 *
 * @brief kenel utils used in minimum spanning tree algorithm.
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
struct GreaterThan
{
    int compare;

    __host__ __device__ __forceinline__
    GreaterThan(int compare) : compare(compare) { }

    __host__ __device__ __forceinline__
    bool operator()(const int &a) const { return (a > compare); }
};

/**
 * @brief Uses the \p d_flags sequence to selectively copy the corresponding 
 * items from \p d_in into \p d_out. The total number of items selected is 
 * stored in \p d_num_selected_out.
 *
 */
template <typename InputT, typename OutputT, typename SizeT, typename FlagT>
cudaError_t CUBSelect_flagged(
    InputT 	*d_in,
    FlagT	*d_flags,
    OutputT	*d_out,
    SizeT	*d_num_selected_out,
    SizeT 	num_elements)
{
    cudaError_t retval = cudaSuccess;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    if(util::GRError(
	    (retval = cub::DeviceSelect::Flagged(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_flags,
		d_out,
		d_num_selected_out,
		num_elements)),
	    "CUBSelect_flagged cub::DeviceSelect::Flagged failed",
	    __FILE__, __LINE__)) return retval;
	
    // allocate temporary storage
    if (util::GRError(
            (retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
            "CUBSelect malloc d_temp_storage failed",
            __FILE__, __LINE__)) return retval;
    
    // run selection
    if (util::GRError(
            (retval = cub::DeviceSelect::Flagged(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_flags,
                d_out,
                d_num_selected_out,
                num_elements)),
            "CUBSelect cub::DeviceSelect::Flagged failed",
            __FILE__, __LINE__)) return retval;

    // clean up
    if (util::GRError(
            (retval = cudaFree(d_temp_storage)),
            "CUBSelect free d_temp_storage failed",
            __FILE__, __LINE__)) return retval;

    return retval;
}

/**
 * @brief selects items from a sequence of int keys using a
 * section functor (greater-than)
 *
 */
template <typename T, typename SizeT>
cudaError_t CUBSelect(
    T             *d_input,
    SizeT         num_elements,
    T	     	  *d_output,
    unsigned int  *num_selected)
{
    cudaError_t retval = cudaSuccess;
    SizeT *d_num_selected = NULL;

    if (util::GRError(
            (retval = cudaMalloc((void**)&d_num_selected, sizeof(SizeT))),
            "CUBSelect d_num_selected malloc failed",
            __FILE__, __LINE__)) return retval;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    GreaterThan select_op(-1);

    // determine temporary device storage requirements
    if (util::GRError(
            (retval = cub::DeviceSelect::If(
                d_temp_storage,
                temp_storage_bytes,
                d_input,
                d_output,
                d_num_selected,
                num_elements,
                select_op)),
            "CUBSelect cub::DeviceSelect::If failed",
            __FILE__, __LINE__)) return retval;

    // allocate temporary storage
    if (util::GRError(
            (retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
            "CUBSelect malloc d_temp_storage failed",
            __FILE__, __LINE__)) return retval;

    // run selection
    if (util::GRError(
            (retval = cub::DeviceSelect::If(
                d_temp_storage,
                temp_storage_bytes,
                d_input,
                d_output,
                d_num_selected,
                num_elements,
                select_op)),
            "CUBSelect cub::DeviceSelect::If failed",
            __FILE__, __LINE__)) return retval;

    if (util::GRError(
            (retval = cudaMemcpy(
                num_selected,
                d_num_selected,
                sizeof(SizeT),
                cudaMemcpyDeviceToHost)),
            "CUBSelect copy back num_selected failed",
            __FILE__, __LINE__)) return retval;

    // clean up
    if (util::GRError(
            (retval = cudaFree(d_temp_storage)),
            "CUBSelect free d_temp_storage failed",
            __FILE__, __LINE__)) return retval;
    if (util::GRError(
            (retval = cudaFree(d_num_selected)),
            "CUBSelect free d_num_selected failed",
            __FILE__, __LINE__)) return retval;

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
