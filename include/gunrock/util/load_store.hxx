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

// Include appropriate CUB library based on compiler
#if defined(__CUDACC__) && !defined(__HIP__)
  // Pure CUDA (NVCC without HIP): Use CUB
  #include <cub/cub.cuh>
  namespace cub_namespace = cub;
  using CacheLoadModifierType = cub::CacheLoadModifier;
  using CacheStoreModifierType = cub::CacheStoreModifier;
  constexpr auto LOAD_DEFAULT = cub::CacheLoadModifier::LOAD_DEFAULT;
  constexpr auto STORE_DEFAULT = cub::CacheStoreModifier::STORE_DEFAULT;
#else
  // HIP compiler (works for both NVIDIA and AMD platforms): Use hipCUB
  // Undefine architecture macros that HIP may define to avoid conflicts with rocPRIM enums
  #ifdef __HIP_PLATFORM_AMD__
    #ifdef gfx942
      #pragma push_macro("gfx942")
      #undef gfx942
    #endif
    #ifdef gfx950
      #pragma push_macro("gfx950")
      #undef gfx950
    #endif
    #ifdef gfx90a
      #pragma push_macro("gfx90a")
      #undef gfx90a
    #endif
  #endif
  #include <hipcub/hipcub.hpp>
  #ifdef __HIP_PLATFORM_AMD__
    #ifdef gfx942
      #pragma pop_macro("gfx942")
    #endif
    #ifdef gfx950
      #pragma pop_macro("gfx950")
    #endif
    #ifdef gfx90a
      #pragma pop_macro("gfx90a")
    #endif
  #endif
  namespace cub_namespace = hipcub;
  using CacheLoadModifierType = hipcub::CacheLoadModifier;
  using CacheStoreModifierType = hipcub::CacheStoreModifier;
  constexpr auto LOAD_DEFAULT = hipcub::CacheLoadModifier::LOAD_DEFAULT;
  constexpr auto STORE_DEFAULT = hipcub::CacheStoreModifier::STORE_DEFAULT;
#endif

namespace gunrock {
namespace thread {

/**
 * @brief Uses a cached load to load a value from a given pointer.
 */
template <CacheLoadModifierType MODIFIER = LOAD_DEFAULT,
          typename type_t>
__device__ __host__ __forceinline__ type_t load(type_t* ptr) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  return cub_namespace::ThreadLoad<MODIFIER>(ptr);
#else
  return *ptr;
#endif
}

/**
 * @brief Uses a cached store to store a given value into a pointer.
 */
template <CacheStoreModifierType MODIFIER = STORE_DEFAULT,
          typename type_t>
__device__ __host__ __forceinline__ void store(type_t* ptr, const type_t& val) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
  cub_namespace::ThreadStore<MODIFIER>(ptr, val);
#else
  *ptr = val;
#endif
}
}  // namespace thread
}  // namespace gunrock
