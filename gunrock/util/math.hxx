/**
 * @file math.hxx
 *
 * @brief
 */

#pragma once

namespace gunrock {

/**
 * @namespace math
 * Math utilities.
 */
namespace math {

/**
 * @brief Statically determine log2(N).
 *
 * @param n
 * @return constexpr int
 */
constexpr int log2(int n) {
  return ((n < 2) ? 1 : 1 + log2(n / 2));
}

namespace atomic {

template <typename type_t>
__host__ __device__ __forceinline__ type_t add(type_t* address,
                                               const type_t& value) {
#ifdef __CUDA_ARCH__
  return atomicAdd(address, value);
#else
  return std::plus<type_t>(*address, value);  // use std::atomic::fetch_add();
#endif
}

template <typename type_t>
__host__ __device__ __forceinline__ type_t min(type_t* address,
                                               const type_t& value) {
#ifdef __CUDA_ARCH__
  return atomicMin(address, value);
#else
  return std::min<type_t>(*address, value);   // use std::atomic;
#endif
}

}  // namespace atomic
}  // namespace math
}  // namespace gunrock