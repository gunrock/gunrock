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

template <typename type_t, typename enable_t = void>
struct numeric_limits : std::numeric_limits<type_t> {};

// Numeric Limits (additional support) for invalid() values.
template <typename type_t>
struct numeric_limits<
    type_t,
    typename std::enable_if_t<std::is_integral<type_t>::value &&
                              std::is_signed<type_t>::value>>
    : std::numeric_limits<type_t> {
  constexpr static type_t invalid() {
    return std::integral_constant<type_t, -1>::value;
  }
};

template <typename type_t>
struct numeric_limits<
    type_t,
    typename std::enable_if_t<std::is_integral<type_t>::value &&
                              std::is_unsigned<type_t>::value>>
    : std::numeric_limits<type_t> {
  constexpr static type_t invalid() {
    return std::integral_constant<type_t,
                                  std::numeric_limits<type_t>::max()>::value;
  }
};

template <typename type_t>
struct numeric_limits<
    type_t,
    typename std::enable_if_t<std::is_floating_point<type_t>::value>>
    : std::numeric_limits<type_t> {
  constexpr static type_t invalid() {
    return std::numeric_limits<type_t>::quiet_NaN();
  }
};

}  // namespace gunrock

namespace gunrock {
namespace util {
namespace limits {

template <typename type_t>
constexpr __host__ __device__ __forceinline__ bool is_valid(type_t value) {
  static_assert((std::is_integral<type_t>::value ||
                 std::is_floating_point<type_t>::value),
                "type_t must be an arithmetic type.");

  // Trying:
  // https://stackoverflow.com/questions/61646166/how-to-resolve-fpclassify-ambiguous-call-to-overloaded-function
  if (std::is_integral<type_t>::value)
    return (value != gunrock::numeric_limits<type_t>::invalid());
  else
    return isnan(static_cast<double>(value)) ? false : true;
}

}  // namespace limits
}  // namespace util
}  // namespace gunrock