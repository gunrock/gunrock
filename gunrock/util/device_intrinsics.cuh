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

//#pragma once

#ifndef DEVICE_INTRINSICS_CUH
#define DEVICE_INTRINSICS_CUH

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/types.cuh>

#define MEMBERMASK 0xffffffff
#define WARPSIZE 32

// CUDA 9 warp shuffles (device intrinsics)
template <typename T>
__device__ static __forceinline__
T _shfl_up(T var, unsigned int delta, int width=WARPSIZE, unsigned mask=MEMBERMASK)
{
  //int first_lane = (WARPSIZE-width) << 8;
#if __CUDACC_VER_MAJOR__ < 9
  return __shfl_up(var, delta, width);
#else
  return __shfl_up_sync(mask, var, delta, width);
#endif
}

template <typename T>
__device__ static __forceinline__
T _shfl_down(T var, unsigned int delta, int width=WARPSIZE, unsigned mask=MEMBERMASK)
{
  //int last_lane = ((WARPSIZE-width) << 8) | 0x1f;
#if __CUDACC_VER_MAJOR__ < 9
  return __shfl_down(var, delta, width);
#else
  return __shfl_down_sync(mask, var, delta, width);
#endif
}

template <typename T>
__device__ static __forceinline__
T _shfl_xor(T var, int lane_mask, int width=WARPSIZE, unsigned mask = MEMBERMASK)
{
  //int last_lane = ((WARPSIZE-width) << 8) | 0x1f;
#if __CUDACC_VER_MAJOR__ < 9
  return __shfl_xor(var, lane_mask, width);
#else
  return __shfl_xor_sync(mask, var, lane_mask, width);
#endif
}

template <typename T>
__device__ static __forceinline__
T _shfl(T var, int source_lane, int width=WARPSIZE, unsigned mask=MEMBERMASK)
{
  //int last_lane = ((WARPSIZE-width) << 8) | 0x1f;
#if __CUDACC_VER_MAJOR__ < 9
  return __shfl(var, source_lane, width);
#else
  return __shfl_sync(mask, var, source_lane, width);
#endif
}

__device__ static __forceinline__
unsigned _ballot(int predicate, unsigned mask=MEMBERMASK)
{
#if __CUDACC_VER_MAJOR__ < 9
  return __ballot(predicate);
#else
  return __ballot_sync(mask, predicate);
#endif
}

__device__ static __forceinline__
int _any(int predicate, unsigned mask=MEMBERMASK)
{
#if __CUDACC_VER_MAJOR__ < 9
  return __any(predicate);
#else
  return __any_sync(mask, predicate);
#endif
}

__device__ static __forceinline__
int _all(int predicate, unsigned mask=MEMBERMASK)
{
#if __CUDACC_VER_MAJOR__ < 9
  return __all(predicate);
#else
  return __all_sync(mask, predicate);
#endif
}

#if __CUDACC_VER_MAJOR__ < 8
// atomic addition from Jon Cohen at NVIDIA
__device__ static __forceinline__ double atomicAdd(double *addr, double val)
{
    double old=*addr, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(
        atomicCAS((unsigned long long int*)addr,
               __double_as_longlong(assumed),
               __double_as_longlong(val + assumed)));
    } while( assumed!=old );
    return old;
}
#endif

__device__ static __forceinline__ long long atomicCAS(long long *addr, long long comp, long long val)
{
    return (long long)atomicCAS(
        (unsigned long long*)addr,
        (unsigned long long )comp,
        (unsigned long long ) val);
}

// TODO: verify overflow condition
__device__ static __forceinline__ long long atomicAdd(long long *addr, long long val)
{
    return (long long)atomicAdd(
        (unsigned long long*)addr,
        (unsigned long long )val);
}

// TODO: only works if both *addr and val are non-negetive
__device__ static __forceinline__ long long atomicMin_(long long* addr, long long val)
{
#if __CUDA_ARCH__ <= 300
    long long pre_value = val;
    long long old_value = val;
    while (true)
    {
        old_value = atomicCAS(addr, pre_value, val);
        if (old_value <= val) break;
        if (old_value == pre_value) break;
        pre_value = old_value;
    }
    return old_value;
#else
    return atomicMin(addr, val);
#endif
}

__device__ static __forceinline__ int atomicMin_(int* addr, int val)
{
    return atomicMin(addr, val);
}

__device__ static __forceinline__ float atomicMin(float* addr, float val)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = ::atomicCAS(addr_as_int, expected, __float_as_int(::fminf(val, __int_as_float(expected))));
    } while (expected != old);
    return __int_as_float(old);
}

template <typename T>
__device__ __forceinline__ T _ldg(T* addr)
{
#if __GR_CUDA_ARCH__ >= 350
    return __ldg(addr);
#else
    return *addr;
#endif
}

namespace gunrock {
namespace util {

/**
 * Terminates the calling thread
 */
__device__ __forceinline__ void ThreadExit() {
    asm("exit;");
}


/**
 * Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
    unsigned int ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}


/**
 * The best way to multiply integers (24 effective bits or less)
 */
__device__ __forceinline__ unsigned int FastMul(unsigned int a, unsigned int b)
{
#if __CUDA_ARCH__ >= 200
    return a * b;
#else
    return __umul24(a, b);
#endif
}


/**
 * The best way to multiply integers (24 effective bits or less)
 */
__device__ __forceinline__ int FastMul(int a, int b)
{
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
struct AtomicInt<T, 4>
{
    static __device__ __forceinline__ T Add(T* ptr, T val)
    {
        return atomicAdd((unsigned int *) ptr, (unsigned int) val);
    }
};

template <typename T>
struct AtomicInt<T, 8>
{
    static __device__ __forceinline__ T Add(T* ptr, T val)
    {
        return atomicAdd((unsigned long long int *) ptr, (unsigned long long int) val);
    }
};

// From Andrew Davidson's dStepping SSSP GPU implementation
// binary search on device, only works for arrays shorter
// than 1024

template <int NT, typename KeyType, typename ArrayType>
__device__ int BinarySearch(KeyType i, ArrayType *queue)
{
    int mid = ((NT >> 1) - 1);

    if (NT > 512)
        mid = queue[mid] > i ? mid - 256 : mid + 256;
    if (NT > 256)
        mid = queue[mid] > i ? mid - 128 : mid + 128;
    if (NT > 128)
        mid = queue[mid] > i ? mid - 64 : mid + 64;
    if (NT > 64)
        mid = queue[mid] > i ? mid - 32 : mid + 32;
    if (NT > 32)
        mid = queue[mid] > i ? mid - 16 : mid + 16;
    mid = queue[mid] > i ? mid - 8 : mid + 8;
    mid = queue[mid] > i ? mid - 4 : mid + 4;
    mid = queue[mid] > i ? mid - 2 : mid + 2;
    mid = queue[mid] > i ? mid - 1 : mid + 1;
    mid = queue[mid] > i ? mid     : mid + 1;

    return mid;
}

} // namespace util
} // namespace gunrock

#endif
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
