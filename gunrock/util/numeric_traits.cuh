// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * numeric_traits.cuh
 *
 * @brief Type traits for numeric types
 */

#pragma once

namespace gunrock {
namespace util {

enum Representation {
  NOT_A_NUMBER,
  SIGNED_INTEGER,
  UNSIGNED_INTEGER,
  FLOATING_POINT
};

template <Representation R>
struct BaseTraits {
  static const Representation REPRESENTATION = R;
};

// Default, non-numeric types
template <typename T>
struct NumericTraits : BaseTraits<NOT_A_NUMBER> {};

template <>
struct NumericTraits<char> : BaseTraits<SIGNED_INTEGER> {};
template <>
struct NumericTraits<signed char> : BaseTraits<SIGNED_INTEGER> {};
template <>
struct NumericTraits<short> : BaseTraits<SIGNED_INTEGER> {};
template <>
struct NumericTraits<int> : BaseTraits<SIGNED_INTEGER> {};
template <>
struct NumericTraits<long> : BaseTraits<SIGNED_INTEGER> {};
template <>
struct NumericTraits<long long> : BaseTraits<SIGNED_INTEGER> {};

template <>
struct NumericTraits<unsigned char> : BaseTraits<UNSIGNED_INTEGER> {};
template <>
struct NumericTraits<unsigned short> : BaseTraits<UNSIGNED_INTEGER> {};
template <>
struct NumericTraits<unsigned int> : BaseTraits<UNSIGNED_INTEGER> {};
template <>
struct NumericTraits<unsigned long> : BaseTraits<UNSIGNED_INTEGER> {};
template <>
struct NumericTraits<unsigned long long> : BaseTraits<UNSIGNED_INTEGER> {};

template <>
struct NumericTraits<float> : BaseTraits<FLOATING_POINT> {};
template <>
struct NumericTraits<double> : BaseTraits<FLOATING_POINT> {};

}  // namespace util
}  // namespace gunrock
