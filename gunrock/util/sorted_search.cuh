// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
* @file
* search_utils.cuh
*
* @brief kernel utils used for search.
*/

#pragma once
#include <cub/cub.cuh>
#include <gunrock/util/array_utils.cuh>
#include <moderngpu/kernel_sortedsearch.hxx>

namespace gunrock {
namespace util {

/**
 * \addtogroup PublicInterface
 * @{
 */

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

template <typename A, typename B, typename C, typename SizeT>
cudaError_t SortedSearch(util::Array1D<SizeT, A> &needles, SizeT num_needles,
                        util::Array1D<SizeT, B> &haystack, SizeT num_haystack,
                        util::Array1D<SizeT, C> &indices,
                        mgpu::context_t *context,
                        bool debug_synchronous = false) {

  if (context == nullptr) {
      return cudaErrorInvalidValue;
  }
  cudaError_t retval = cudaSuccess;

  // TODO: Experiment with these values and choose the best ones. We
  // could even choose to make them a parameter of util::SortedSearch
  // and choose based on our usage
  typedef mgpu::launch_box_t<
      mgpu::arch_30_cta<256, 7>,
      mgpu::arch_35_cta<256, 7>,
      mgpu::arch_50_cta<256, 7>,
      mgpu::arch_60_cta<256, 7>,
      mgpu::arch_70_cta<256, 7>,
      mgpu::arch_75_cta<256, 7>
      > launch_t;

  mgpu::sorted_search<mgpu::bounds_lower, launch_t>(needles.GetPointer(util::DEVICE), num_needles,
                                                    haystack.GetPointer(util::DEVICE), num_haystack,
                                                    indices.GetPointer(util::DEVICE), mgpu::less_t<A>(),
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
