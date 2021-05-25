/**
 * @file load_store.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-25
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <cub/cub.cuh>

namespace gunrock {
namespace thread {

/**
 * @brief Uses a cached load to load a value from a given pointer.
 */
template <
    cub::CacheLoadModifier MODIFIER = cub::CacheLoadModifier::LOAD_DEFAULT,
    typename type_t>
__device__ __host__ __forceinline__ type_t load(type_t* ptr) {
#ifdef __CUDA_ARCH__
  return cub::ThreadLoad<MODIFIER>(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Uses a cached store to store a given value into a pointer.
 */
template <
    cub::CacheStoreModifier MODIFIER = cub::CacheStoreModifier::STORE_DEFAULT,
    typename type_t>
__device__ __host__ __forceinline__ void store(type_t* ptr, const type_t& val) {
#ifdef __CUDA_ARCH__
  cub::ThreadStore<MODIFIER>(ptr, val);
#else
  *ptr = val;
#endif
}
}  // namespace thread
}  // namespace gunrock