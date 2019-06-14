// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cub_io.cuh
 *
 * @brief Warpers to use cub io functions
 */

#pragma once

#include <cub/cub.cuh>

namespace gunrock {

template <
    cub::CacheLoadModifier MODIFIER = cub::CacheLoadModifier::LOAD_DEFAULT,
    typename T>
__device__ __host__ __forceinline__ T Load(T* ptr) {
#ifdef __CUDA_ARCH__
  return cub::ThreadLoad<MODIFIER>(ptr);
#else
  return *ptr;
#endif
}

template <
    cub::CacheStoreModifier MODIFIER = cub::CacheStoreModifier::STORE_DEFAULT,
    typename T>
__device__ __host__ __forceinline__ void Store(T* ptr, const T& val) {
#ifdef __CUDA_ARCH__
  cub::ThreadStore<MODIFIER>(ptr, val);
#else
  *ptr = val;
#endif
}

}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
