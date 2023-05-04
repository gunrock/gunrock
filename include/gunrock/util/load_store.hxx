/**
 * @file load_store.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2021-05-25
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
#include <hipcub/hipcub.hpp>

namespace gunrock {
namespace thread {

/**
 * @brief Uses a cached load to load a value from a given pointer.
 */
template <hipcub::CacheLoadModifier MODIFIER =
              hipcub::CacheLoadModifier::LOAD_DEFAULT,
          typename type_t>
__device__ __host__ __forceinline__ type_t load(type_t* ptr) {
#ifdef __HIP_DEVICE_COMPILE__
  // #ifdef __CUDA_ARCH__
  return hipcub::ThreadLoad<MODIFIER>(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Uses a cached store to store a given value into a pointer.
 */
template <hipcub::CacheStoreModifier MODIFIER =
              hipcub::CacheStoreModifier::STORE_DEFAULT,
          typename type_t>
__device__ __host__ __forceinline__ void store(type_t* ptr, const type_t& val) {
#ifdef __HIP_DEVICE_COMPILE__
  // #ifdef __CUDA_ARCH__
  hipcub::ThreadStore<MODIFIER>(ptr, val);
#else
  *ptr = val;
#endif
}
}  // namespace thread
}  // namespace gunrock
