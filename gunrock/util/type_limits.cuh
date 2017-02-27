// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * type_limits.cuh
 *
 * @brief data types and limits defination
 */

#pragma once
#include <limits>

namespace gunrock {
namespace util {

// Ensure no un-specialized types will be compiled
extern __device__ __host__
void Error_UnsupportedType();

template <typename T>
struct PreDefinedValues{};

template <>
struct PreDefinedValues<char>
{
    static const char MinValue = CHAR_MIN;
    static const char MaxValue = CHAR_MAX;
    static const char AllZeros = (char)0;
    static const char AllOnes  = ~AllZeros;
    static const char InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<unsigned char>
{
    static const unsigned char MinValue = (unsigned char)0;
    static const unsigned char MaxValue = UCHAR_MAX;
    static const unsigned char AllZeros = (unsigned char)0;
    static const unsigned char AllOnes  = ~AllZeros;
    static const unsigned char InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<signed char>
{
    static const signed char MinValue = SCHAR_MIN;
    static const signed char MaxValue = SCHAR_MAX;
    static const signed char AllZeros = (signed char)0;
    static const signed char AllOnes  = ~AllZeros;
    static const signed char InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<short>
{
    static const short MinValue = SHRT_MIN;
    static const short MaxValue = SHRT_MAX;
    static const short AllZeros = (short)0;
    static const short AllOnes  = ~AllZeros;
    static const short InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<unsigned short>
{
    static const unsigned short MinValue = (unsigned short)0;
    static const unsigned short MaxValue = USHRT_MAX;
    static const unsigned short AllZeros = (unsigned short)0;
    static const unsigned short AllOnes  = ~AllZeros;
    static const unsigned short InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<int>
{
    static const int MinValue = INT_MIN;
    static const int MaxValue = INT_MAX;
    static const int AllZeros = (int)0;
    static const int AllOnes  = ~AllZeros;
    static const int InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<unsigned int>
{
    static const unsigned int MinValue = (unsigned int)0;
    static const unsigned int MaxValue = UINT_MAX;
    static const unsigned int AllZeros = (unsigned int)0;
    static const unsigned int AllOnes  = ~AllZeros;
    static const unsigned int InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<long>
{
    static const long MinValue = LONG_MIN;
    static const long MaxValue = LONG_MAX;
    static const long AllZeros = (long)0;
    static const long AllOnes  = ~AllZeros;
    static const long InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<unsigned long>
{
    static const unsigned long MinValue = (unsigned long)0;
    static const unsigned long MaxValue = ULONG_MAX;
    static const unsigned long AllZeros = (unsigned long)0;
    static const unsigned long AllOnes  = ~AllZeros;
    static const unsigned long InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<long long>
{
    static const long long MinValue = LLONG_MIN;
    static const long long MaxValue = LLONG_MAX;
    static const long long AllZeros = (long long)0;
    static const long long AllOnes  = ~AllZeros;
    static const long long InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<unsigned long long>
{
    static const unsigned long long MinValue = (unsigned long long)0;
    static const unsigned long long MaxValue = ULLONG_MAX;
    static const unsigned long long AllZeros = (unsigned long long)0;
    static const unsigned long long AllOnes  = ~AllZeros;
    static const unsigned long long InvalidValue = AllOnes;
};

template <typename T>
__device__ __host__ __forceinline__
bool isValid(T val)
{
    return (val != PreDefinedValues<T>::InvalidValue);
}

} // namespace util
} // namespace gunrock



// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
