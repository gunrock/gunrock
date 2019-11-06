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
#include <cfloat>

namespace gunrock {
namespace util {

// Ensure no un-specialized types will be compiled
extern __device__ __host__ void Error_UnsupportedType();

template <typename T>
struct PreDefinedValues {};

template <>
struct PreDefinedValues<char> {
  static const char MinValue = CHAR_MIN;
  static const char MaxValue = CHAR_MAX;
  static const char AllZeros = (char)0;
  static const char AllOnes = ~AllZeros;
  static const char InvalidValue = AllOnes;
  static const bool Signed = true;
  typedef short PromoteType;
};

template <>
struct PreDefinedValues<unsigned char> {
  static const unsigned char MinValue = (unsigned char)0;
  static const unsigned char MaxValue = UCHAR_MAX;
  static const unsigned char AllZeros = (unsigned char)0;
  static const unsigned char AllOnes = ~AllZeros;
  static const unsigned char InvalidValue = AllOnes;
  static const bool Signed = false;
  typedef unsigned short PromoteType;
};

template <>
struct PreDefinedValues<signed char> {
  static const signed char MinValue = SCHAR_MIN;
  static const signed char MaxValue = SCHAR_MAX;
  static const signed char AllZeros = (signed char)0;
  static const signed char AllOnes = ~AllZeros;
  static const signed char InvalidValue = AllOnes;
  static const bool Signed = true;
  typedef signed short PromoteType;
};

template <>
struct PreDefinedValues<short> {
  static const short MinValue = SHRT_MIN;
  static const short MaxValue = SHRT_MAX;
  static const short AllZeros = (short)0;
  static const short AllOnes = ~AllZeros;
  static const short InvalidValue = AllOnes;
  static const bool Signed = true;
  typedef int PromoteType;
  typedef char DemoteType;
};

template <>
struct PreDefinedValues<unsigned short> {
  static const unsigned short MinValue = (unsigned short)0;
  static const unsigned short MaxValue = USHRT_MAX;
  static const unsigned short AllZeros = (unsigned short)0;
  static const unsigned short AllOnes = ~AllZeros;
  static const unsigned short InvalidValue = AllOnes;
  static const bool Signed = false;
  typedef unsigned int PromoteType;
  typedef unsigned char DemoteType;
};

template <>
struct PreDefinedValues<int> {
  static const int MinValue = INT_MIN;
  static const int MaxValue = INT_MAX;
  static const int AllZeros = (int)0;
  static const int AllOnes = ~AllZeros;
  static const int InvalidValue = AllOnes;
  static const bool Signed = true;
  typedef short DemoteType;
#if LONG_MAX == INT_MAX
  typedef long long PromoteType;
#else
  typedef long PromoteType;
#endif
};

template <>
struct PreDefinedValues<unsigned int> {
  static const unsigned int MinValue = (unsigned int)0;
  static const unsigned int MaxValue = UINT_MAX;
  static const unsigned int AllZeros = (unsigned int)0;
  static const unsigned int AllOnes = ~AllZeros;
  static const unsigned int InvalidValue = AllOnes;
  static const bool Signed = false;
  typedef unsigned short DemoteType;
#if LONG_MAX == INT_MAX
  typedef unsigned long long PromoteType;
#else
  typedef unsigned long PromoteType;
#endif
};

template <>
struct PreDefinedValues<long> {
  static const long MinValue = LONG_MIN;
  static const long MaxValue = LONG_MAX;
  static const long AllZeros = (long)0;
  static const long AllOnes = ~AllZeros;
  static const long InvalidValue = AllOnes;
  static const bool Signed = true;
  typedef long long PromoteType;
#if LONG_MAX == INT_MAX
  typedef short DemoteType;
#else
  typedef int DemoteType;
#endif
};

template <>
struct PreDefinedValues<unsigned long> {
  static const unsigned long MinValue = (unsigned long)0;
  static const unsigned long MaxValue = ULONG_MAX;
  static const unsigned long AllZeros = (unsigned long)0;
  static const unsigned long AllOnes = ~AllZeros;
  static const unsigned long InvalidValue = AllOnes;
  static const bool Signed = false;
  typedef unsigned long long PromoteType;
#if LONG_MAX == INT_MAX
  typedef unsigned short DemoteType;
#else
  typedef unsigned int DemoteType;
#endif
};

template <>
struct PreDefinedValues<long long> {
  static const long long MinValue = LLONG_MIN;
  static const long long MaxValue = LLONG_MAX;
  static const long long AllZeros = (long long)0;
  static const long long AllOnes = ~AllZeros;
  static const long long InvalidValue = AllOnes;
  static const bool Signed = true;
  typedef long DemoteType;
};

template <>
struct PreDefinedValues<unsigned long long> {
  static const unsigned long long MinValue = (unsigned long long)0;
  static const unsigned long long MaxValue = ULLONG_MAX;
  static const unsigned long long AllZeros = (unsigned long long)0;
  static const unsigned long long AllOnes = ~AllZeros;
  static const unsigned long long InvalidValue = AllOnes;
  static const bool Signed = false;
  typedef unsigned long DemoteType;
};

template <>
struct PreDefinedValues<float> {
  constexpr static const float MinValue = FLT_MIN;
  constexpr static const float MaxValue = FLT_MAX;
  constexpr static const float InvalidValue = NAN;
  static const bool Signed = true;
  typedef double PromoteType;
};

template <>
struct PreDefinedValues<double> {
  constexpr static const double MinValue = DBL_MIN;
  constexpr static const double MaxValue = DBL_MAX;
  constexpr static const double InvalidValue = NAN;
  static const bool Signed = true;
  typedef long double PromoteType;
  typedef float DemoteType;
};

template <>
struct PreDefinedValues<long double> {
  constexpr static const long double MinValue = LDBL_MIN;
  constexpr static const long double MaxValue = LDBL_MAX;
  constexpr static const long double InvalidValue = NAN;
  static const bool Signed = true;
  typedef double DemoteType;
};

template <typename T>
__device__ __host__ __forceinline__ bool isValid(const T &val) {
  return (val != PreDefinedValues<T>::InvalidValue);
}

template <>
__device__ __host__ __forceinline__ bool isValid(const float &val) {
  return (!isnan(val));
}

template <>
__device__ __host__ __forceinline__ bool isValid(const double &val) {
  return (!isnan(val));
}

template <>
__device__ __host__ __forceinline__ bool isValid(const long double &val) {
  return (!isnan(val));
}

template <typename T, bool SIGNED>
struct Switch_Signed {
  __device__ __host__ __forceinline__ static bool lessThanZero(const T &val) {
    return val < 0;
  }

  __device__ __host__ __forceinline__ static bool atLeastZero(const T &val) {
    return val >= 0;
  }
};

template <typename T>
struct Switch_Signed<T, false> {
  __device__ __host__ __forceinline__ static bool lessThanZero(const T &val) {
    return false;
  }

  __device__ __host__ __forceinline__ static bool atLeastZero(const T &val) {
    return true;
  }
};

template <typename T>
__device__ __host__ __forceinline__ bool lessThanZero(const T &val) {
  return Switch_Signed<T, PreDefinedValues<T>::Signed>::lessThanZero(val);
}

template <typename T>
__device__ __host__ __forceinline__ bool atLeastZero(const T &val) {
  return Switch_Signed<T, PreDefinedValues<T>::Signed>::atLeastZero(val);
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
