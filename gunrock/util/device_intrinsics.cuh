// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * device_intrinsics.cuh
 *
 * @brief Common device intrinsics (potentially specialized by architecture)
 */

#pragma once

#include <limits>
#include <atomic>
#include <gunrock/util/cuda_properties.cuh>
//#include <gunrock/util/types.cuh>

#ifndef MEMBERBASK
#define MEMBERMASK 0xffffffffu
#endif

#ifndef MEMBERMASK
#define MEMBERMASK 0xffffffffu
#endif

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#if (__CUDACC_VER_MAJOR__ >= 9 && __CUDA_ARCH__ >= 300) && \
    !defined(USE_SHFL_SYNC)
#define USE_SHFL_SYNC
#endif

// CUDA 9 warp shuffles (device intrinsics)
template <typename T>
__device__ static __forceinline__ T _shfl_up(T var, unsigned int delta,
                                             int width = WARPSIZE,
                                             unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  var = __shfl_up_sync(mask, var, delta, width);
#else
#if (__CUDA_ARCH__ >= 300)
  var = __shfl_up(var, delta, width);
#endif
#endif
  return var;
}

template <typename T>
__device__ static __forceinline__ T _shfl_down(T var, unsigned int delta,
                                               int width = WARPSIZE,
                                               unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  var = __shfl_down_sync(mask, var, delta, width);
#else
#if (__CUDA_ARCH__ >= 300)
  var = __shfl_down(var, delta, width);
#endif
#endif
  return var;
}

template <typename T>
__device__ static __forceinline__ T _shfl_xor(T var, int lane_mask,
                                              int width = WARPSIZE,
                                              unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  var = __shfl_xor_sync(mask, var, lane_mask, width);
#else
#if (__CUDA_ARCH__ >= 300)
  var = __shfl_xor(var, lane_mask, width);
#endif
#endif
  return var;
}

template <typename T>
__device__ static __forceinline__ T _shfl(T var, int source_lane,
                                          int width = WARPSIZE,
                                          unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  var = __shfl_sync(mask, var, source_lane, width);
#else
#if (__CUDA_ARCH__ >= 300)
  var = __shfl(var, source_lane, width);
#endif
#endif
  return var;
}

__device__ static __forceinline__ unsigned _ballot(int predicate,
                                                   unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  return __ballot_sync(mask, predicate);
#else
#if (__CUDA_ARCH__ >= 300)
  return __ballot(predicate);
#endif
#endif
}

__device__ static __forceinline__ int _any(int predicate,
                                           unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  return __any_sync(mask, predicate);
#else
#if (__CUDA_ARCH__ >= 300)
  return __any(predicate);
#endif
#endif
}

__device__ static __forceinline__ int _all(int predicate,
                                           unsigned mask = MEMBERMASK) {
#ifdef USE_SHFL_SYNC
  return __all_sync(mask, predicate);
#else
#if (__CUDA_ARCH__ >= 300)
  return __all(predicate);
#endif
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// atomic addition from Jon Cohen at NVIDIA
__device__ static double atomicAdd(double* addr, double val) {
  double old = *addr, assumed;
  do {
    assumed = old;
    old = __longlong_as_double(atomicCAS((unsigned long long int*)addr,
                                         __double_as_longlong(assumed),
                                         __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}
#else
#endif

__device__ static long long atomicCAS(long long* addr, long long comp,
                                      long long val) {
  return (long long)atomicCAS((unsigned long long*)addr,
                              (unsigned long long)comp,
                              (unsigned long long)val);
}

__device__ static float atomicCAS(float* addr, float comp, float val) {
  return __int_as_float(
      atomicCAS((int*)addr, __float_as_int(comp), __float_as_int(val)));
}

__device__ static double atomicCAS(double* addr, double comp, double val) {
  return __longlong_as_double(atomicCAS(
      (long long*)addr, __double_as_longlong(comp), __double_as_longlong(val)));
}

// TODO: verify overflow condition
__device__ static long long atomicAdd(long long* addr, long long val) {
  return (long long)atomicAdd((unsigned long long*)addr,
                              (unsigned long long)val);
}

#if ULONG_MAX == ULLONG_MAX
__device__ static unsigned long atomicAdd(unsigned long* addr,
                                          unsigned long val) {
  return (unsigned long long)atomicAdd((unsigned long long*)addr,
                                       (unsigned long long)val);
}
#endif

#if __GR_CUDA_ARCH__ <= 300
// TODO: only works if both *addr and val are non-negetive
/*__device__ static signed long long int atomicMin(signed long long int* addr,
signed long long int val)
{
    unsigned long long int pre_value = (unsigned long long int)val;
    unsigned long long int old_value = (unsigned long long int)val;
    while (true)
    {
        old_value = atomicCAS((unsigned long long int*)addr, pre_value,
(unsigned long long int)val); if (old_value <= (unsigned long long int)val)
break; if (old_value == pre_value) break; pre_value = old_value;
    }
    return old_value;
}*/
#endif

//#if UINT64_MAX != ULLONG_MAX
__device__ static uint64_t atomicMin(uint64_t* addr, uint64_t val) {
  return (uint64_t)atomicMin((unsigned long long int*)addr,
                             (unsigned long long int)val);
  //    unsigned long long int old = (unsigned long long int)(*addr);
  //    unsigned long long int expected;
  //    do {
  //        expected = old;
  //        old = atomicCAS(
  //            (unsigned long long int*)addr,
  //            expected, min((unsigned long long int)val, expected));
  //    } while (expected != old);
  //    return old;
}
//#endif

__device__ static float atomicMin(float* addr, float val) {
  int* addr_as_int = (int*)addr;
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = ::atomicCAS(addr_as_int, expected,
                      __float_as_int(::fminf(val, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

__device__ static double atomicMin(double* addr, double val) {
  long long* addr_as_longlong = (long long*)addr;
  long long old = *addr_as_longlong;
  long long expected;
  do {
    expected = old;
    old = ::atomicCAS(
        addr_as_longlong, expected,
        __double_as_longlong(::fmin(val, __longlong_as_double(expected))));
  } while (expected != old);
  return __longlong_as_double(old);
}

__device__ static float atomicMax(float* addr, float val) {
  int* addr_as_int = (int*)addr;
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = ::atomicCAS(addr_as_int, expected,
                      __float_as_int(::fmaxf(val, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

__device__ static double atomicMax(double* addr, double val) {
  long long* addr_as_longlong = (long long*)addr;
  long long old = *addr_as_longlong;
  long long expected;
  do {
    expected = old;
    old = ::atomicCAS(
        addr_as_longlong, expected,
        __double_as_longlong(::fmax(val, __longlong_as_double(expected))));
  } while (expected != old);
  return __longlong_as_double(old);
}

template <typename T>
__device__ __host__ __forceinline__ T _ldg(T* addr) {
#ifdef __CUDA_ARCH__
#if __GR_CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
#else
  return *addr;
#endif
}

template <typename T>
__device__ __host__ __forceinline__ T _atomicAdd(T* ptr, const T& val) {
#ifdef __CUDA_ARCH__
  return atomicAdd(ptr, val);
#else
  T retval;
#pragma omp atomic capture
  {
    retval = ptr[0];
    ptr[0] += val;
  }
  return retval;
#endif
}

template <typename T>
__device__ __host__ __forceinline__ T _atomicMin(T* ptr, const T& val) {
#ifdef __CUDA_ARCH__
  return atomicMin(ptr, val);
#else
  std::atomic<T>* atomic_ptr = reinterpret_cast<std::atomic<T>*>(ptr);

  T old_val = *ptr;
  while (true) {
    bool is_equal = std::atomic_compare_exchange_strong(atomic_ptr, &old_val,
                                                        min(old_val, val));
    if (is_equal) break;
  }
  return old_val;
#endif
}

namespace gunrock {
namespace util {

/**
 * Terminates the calling thread
 */
__device__ __forceinline__ void ThreadExit() { asm("exit;"); }

/**
 * Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId() {
  unsigned int ret;
  asm("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

/**
 * The best way to multiply integers (24 effective bits or less)
 */
__device__ __forceinline__ unsigned int FastMul(unsigned int a,
                                                unsigned int b) {
#if __CUDA_ARCH__ >= 200
  return a * b;
#else
  return __umul24(a, b);
#endif
}

/**
 * The best way to multiply integers (24 effective bits or less)
 */
__device__ __forceinline__ int FastMul(int a, int b) {
#if __CUDA_ARCH__ >= 200
  return a * b;
#else
  return __mul24(a, b);
#endif
}

/**
 * Wrapper for performing atomic operations on integers of type size_t
 */
template <typename T, int SizeT = sizeof(T)>
struct AtomicInt;

template <typename T>
struct AtomicInt<T, 4> {
  static __device__ __forceinline__ T Add(T* ptr, T val) {
    return atomicAdd((unsigned int*)ptr, (unsigned int)val);
  }
};

template <typename T>
struct AtomicInt<T, 8> {
  static __device__ __forceinline__ T Add(T* ptr, T val) {
    return atomicAdd((unsigned long long int*)ptr, (unsigned long long int)val);
  }
};

// From Andrew Davidson's dStepping SSSP GPU implementation
// binary search on device, only works for arrays shorter
// than 1024

template <int NT, typename KeyType, typename ArrayType>
__device__ int BinarySearch(KeyType i, ArrayType* queue) {
  int mid = ((NT >> 1) - 1);

  if (NT > 512) mid = queue[mid] > i ? mid - 256 : mid + 256;
  if (NT > 256) mid = queue[mid] > i ? mid - 128 : mid + 128;
  if (NT > 128) mid = queue[mid] > i ? mid - 64 : mid + 64;
  if (NT > 64) mid = queue[mid] > i ? mid - 32 : mid + 32;
  if (NT > 32) mid = queue[mid] > i ? mid - 16 : mid + 16;
  mid = queue[mid] > i ? mid - 8 : mid + 8;
  mid = queue[mid] > i ? mid - 4 : mid + 4;
  mid = queue[mid] > i ? mid - 2 : mid + 2;
  mid = queue[mid] > i ? mid - 1 : mid + 1;
  mid = queue[mid] > i ? mid : mid + 1;

  return mid;
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
