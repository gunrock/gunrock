/**
 * @file atomic_functions.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once
namespace gunrock {
namespace gcuda {

/**
 * @brief Wrapper around CUDA's natively supported atomicMin types.
 *
 * @tparam type_t
 * @param address
 * @param value
 * @return type_t
 */
template <typename type_t>
__device__ static type_t atomicMin(type_t* address, type_t value) {
  return ::atomicMin(address, value);
}

/**
 * @brief CUDA natively doesn't support atomicMin on float based addresses and
 * values. This is a workaround (as of CUDA 11.1, there've been no support).
 *
 * @param address
 * @param value
 * @return float
 */
__device__ static float atomicMin(float* address, float value) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = ::atomicCAS(addr_as_int, expected,
                      __float_as_int(::fminf(value, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

/**
 * @brief CUDA natively doesn't support atomicMin on double based addresses and
 * values. This is a workaround (as of CUDA 11.1, there've been no support).
 *
 * @param address
 * @param value
 * @return double
 */
__device__ static double atomicMin(double* address, double value) {
  unsigned long long* addr_as_longlong =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *addr_as_longlong;
  unsigned long long expected;
  do {
    expected = old;
    old = ::atomicCAS(
        addr_as_longlong, expected,
        __double_as_longlong(::fmin(value, __longlong_as_double(expected))));
  } while (expected != old);
  return __longlong_as_double(old);
}

/**
 * @brief Wrapper around CUDA's natively supported atomicMax types.
 *
 * @tparam type_t
 * @param address
 * @param value
 * @return __device__
 */
template <typename type_t>
__device__ static type_t atomicMax(type_t* address, type_t value) {
  return ::atomicMax(address, value);
}

/**
 * @brief CUDA natively doesn't support atomicMax on float based addresses and
 * values. This is a workaround (as of CUDA 11.1, there've been no support).
 *
 * @param address
 * @param value
 * @return float
 */
__device__ static float atomicMax(float* address, float val) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = ::atomicCAS(addr_as_int, expected,
                      __float_as_int(::fmaxf(val, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

/**
 * @brief CUDA natively doesn't support atomicMax on double based addresses and
 * values. This is a workaround (as of CUDA 11.1, there've been no support).
 *
 * @param address
 * @param value
 * @return double
 */
__device__ static double atomicMax(double* address, double value) {
  unsigned long long* addr_as_longlong =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *addr_as_longlong;
  unsigned long long expected;
  do {
    expected = old;
    old = ::atomicCAS(
        addr_as_longlong, expected,
        __double_as_longlong(::fmax(value, __longlong_as_double(expected))));
  } while (expected != old);
  return __longlong_as_double(old);
}

}  // namespace gcuda
}  // namespace gunrock