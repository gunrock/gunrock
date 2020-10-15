/**
 * @file limits.hxx
 *
 * @brief
 */

#pragma once

#include <limits>
#include <cmath>

#include <gunrock/util/type_traits.hxx>

namespace gunrock {

template <typename type_t>
struct numeric_limits : std::numeric_limits<type_t> {};

// Numeric Limits (additional support) for invalid() values.
template <>
struct numeric_limits<int> : std::numeric_limits<int> {
  // XXX: This doesn't seem like the right thing to do for a signed integral
  // type. It implies, -1 is an invalid value. Integers do not have a notion of
  // invalid values.
  constexpr static int invalid() { return ~((int)0); }
};

template <>
struct numeric_limits<float> : std::numeric_limits<float> {
  constexpr static float invalid() {
    return std::numeric_limits<float>::quiet_NaN();
  }
};

template <>
struct numeric_limits<double> : std::numeric_limits<double> {
  constexpr static double invalid() {
    return std::numeric_limits<double>::quiet_NaN();
  }
};

template <>
struct numeric_limits<long double> : std::numeric_limits<long double> {
  constexpr static long double invalid() {
    return std::numeric_limits<long double>::quiet_NaN();
  }
};

}  // namespace gunrock

namespace gunrock {
namespace util {
namespace limits {

// XXX: Revisit later...
template <typename numeric_t>
__host__ __device__ __forceinline__ bool is_valid(numeric_t const& value) {
  static_assert(std::is_arithmetic_v<numeric_t>);
  return !isnan(value);
}

template <typename numeric_t>
__host__ __device__ __forceinline__ bool is_invalid(numeric_t const& value) {
  static_assert(std::is_arithmetic_v<numeric_t>);
  return isnan(value);
}

}  // namespace limits
}  // namespace util
}  // namespace gunrock