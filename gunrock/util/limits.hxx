/**
 * @file limits.hxx
 *
 * @brief
 */

#pragma once

namespace gunrock {
namespace util {

/**
 * @namespace limits
 * Min, max, etc. utilities.
 */
namespace limits {

/**
 * @brief max Given a, b of type_t find: ((a > b) ? a : b) at compile time. Starting
 * C++14, std::max have support for constexpr, so we just use that.
 *
 * @tparam type_t type of value to be compared.
 * @param a l.h.s of max comparison
 * @param b r.h.s of max comparison
 * @return constexpr const type_t& the maximum value between a and b.
 */
template<typename type_t>
constexpr const type_t&
max(const type_t& a, const type_t& b)
{
  return std::max(a, b);
}

/**
 * @brief min Given a, b of type_t find: ((a < b) ? a : b) at compile time. Starting
 * C++14, std::min have support for constexpr, so we just use that.
 *
 * @tparam type_t type of value to be compared.
 * @param a l.h.s of min comparison
 * @param b r.h.s of min comparison
 * @return constexpr const type_t& the minimum value between a and b.
 */
template<typename type_t>
constexpr const type_t&
min(const type_t& a, const type_t& b)
{
  return std::min(a, b);
}

} // namespace limits

} // namespace util
} // namespace gunrock