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

#include <gunrock/util/cuda_properties.cuh>

// atomic addition from Jon Cohen at NVIDIA
__device__ static double atomicAdd(double *addr, double val)
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

__device__ static long long atomicCAS(long long *addr, long long comp, long long val)
{
    return (long long)atomicCAS(
        (unsigned long long*)addr,
        (unsigned long long )comp,
        (unsigned long long ) val);
}

// TODO: verify overflow condition
__device__ static long long atomicAdd(long long *addr, long long val)
{
    return (long long)atomicAdd(
        (unsigned long long*)addr,
        (unsigned long long )val);
}

// TODO: only works if both *addr and val are non-negetive
//__device__ static long long atomicMin(long long *addr, long long val)
//{
//    return (long long)atomicMin(
//        (unsigned long long*)addr,
//        (unsigned long long )val);
//}

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

// Ensure no un-specialized types will be compiled
extern __device__ __host__ void Error_UnsupportedType();

template <typename T>
__device__ __host__ __forceinline__ T MaxValue()
{
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int MaxValue<int>()
{
    return INT_MAX;
}

template <>
__device__ __host__ __forceinline__ long long MaxValue<long long>()
{
    return LLONG_MAX;
}

template <typename T>
__device__ __host__ __forceinline__ T MinValue()
{
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int MinValue<int>()
{
    return INT_MIN;
}

template <>
__device__ __host__ __forceinline__ long long MinValue<long long>()
{
    return LLONG_MIN;
}

/*template <typename T, size_t SIZE>
__device__ __host__ __forceinline__ T AllZeros_N()
{
    Error_UnsupportedSize();
    return 0;
}

template <typename T>
__device__ __host__ __forceinline__ T AllZeros_N<T, 4>()
{
    return (T)0x00000000;
}

template <typename T>
__device__ __host__ __forceinline__ T AllZeros_N<T, 8>()
{
    return (T)0x0000000000000000;
}*/

template <typename T>
__device__ __host__ __forceinline__ T AllZeros()
{
    //return AllZeros_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int AllZeros<int>()
{
    return (int)0x00000000;
}

template <>
__device__ __host__ __forceinline__ long long AllZeros<long long>()
{
    return (long long)0x0000000000000000LL;
}


/*template <typename T, size_t SIZE>
__device__ __host__ __forceinline__ T AllOnes_N()
{
    Error_UnsupportedSize();
    return 0;
}

template <typename T>
__device__ __host__ __forceinline__ T AllOnes_N<T, 4>()
{
    return (T)0xFFFFFFFF;
}

template <typename T>
__device__ __host__ __forceinline__ T AllOnes_N<T, 8>()
{
    return (T)0xFFFFFFFFFFFFFFFF;
}*/

template <typename T>
__device__ __host__ __forceinline__ T AllOnes()
{
    //return AllOnes_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int AllOnes<int>()
{
    return (int)0xFFFFFFFF;
}

template <>
__device__ __host__ __forceinline__ long long AllOnes<long long>()
{
    return (long long)0xFFFFFFFFFFFFFFFFLL;
}

template <typename T>
__device__ __host__ __forceinline__ T InvalidValue()
{
    //return AllOnes_N<T, sizeof(T)>();
    Error_UnsupportedType();
    return 0;
}

template <>
__device__ __host__ __forceinline__ int InvalidValue<int>()
{
    return (int)-1;
}

template <>
__device__ __host__ __forceinline__ long long InvalidValue<long long>()
{
    return (long long)-1;
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


} // namespace util
} // namespace gunrock
