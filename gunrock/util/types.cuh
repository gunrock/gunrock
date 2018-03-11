// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * types.cuh
 *
 * @brief data types and limits defination
 */

#pragma once

#include <float.h>

namespace gunrock {
namespace util {

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
__device__ __host__ __forceinline__ float MaxValue<float>()
{
    return FLT_MAX;
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
__device__ __host__ __forceinline__ float MinValue<float>()
{
    return FLT_MIN;
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

template <>
__device__ __host__ __forceinline__ unsigned char AllOnes<unsigned char>()
{
    return (unsigned char)0xFF;
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

template <typename T>
__device__ __host__ __forceinline__ bool isValid(T val)
{
    return val >= 0;//(val != InvalidValue<T>());
}

template <typename T, int SIZE>
struct VectorType{ /*typedef UnknownType Type;*/};
template <> struct VectorType<int      , 1> {typedef int       Type;};
template <> struct VectorType<int      , 2> {typedef int2      Type;};
template <> struct VectorType<int      , 3> {typedef int3      Type;};
template <> struct VectorType<int      , 4> {typedef int4      Type;};
template <> struct VectorType<long long, 1> {typedef long long Type;};
template <> struct VectorType<long long, 2> {typedef longlong2 Type;};
template <> struct VectorType<long long, 3> {typedef longlong3 Type;};
template <> struct VectorType<long long, 4> {typedef longlong4 Type;};

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
