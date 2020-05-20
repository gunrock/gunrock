/**
 * @file limits.hxx
 *
 * @brief
 */

#pragma once

namespace gunrock {
namespace util {

/**
 * @namespace math
 * Math utilities.
 */
namespace math {

/**
 * @brief Statically determine log2(N).
 *
 * @param n
 * @return constexpr int
 */
constexpr int
log2(int n)
{
  return ((n < 2) ? 1 : 1 + log2(n / 2));
}

} // namespace math

} // namespace util
} // namespace gunrock