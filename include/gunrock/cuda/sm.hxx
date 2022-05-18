/**
 * @file sm.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-17
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

namespace gunrock {
namespace gcuda {

namespace launch_box {

/**
 * @brief Bit flag enum representing different SM architectures. `fallback` is a
 * bit vector of all 1's so it can be used to represent all SM architectures.
 *
 */
enum sm_flag_t : unsigned {
  fallback = ~0u,
  sm_30 = 1 << 0,
  sm_35 = 1 << 1,
  sm_37 = 1 << 2,
  sm_50 = 1 << 3,
  sm_52 = 1 << 4,
  sm_53 = 1 << 5,
  sm_60 = 1 << 6,
  sm_61 = 1 << 7,
  sm_62 = 1 << 8,
  sm_70 = 1 << 9,
  sm_72 = 1 << 10,
  sm_75 = 1 << 11,
  sm_80 = 1 << 12,
  sm_86 = 1 << 13
};

/**
 * @brief Overloaded bitwise OR operator.
 *
 * @param lhs Left-hand side.
 * @param rhs Right-hand side.
 * \return sm_flag_t
 */
constexpr sm_flag_t operator|(sm_flag_t lhs, sm_flag_t rhs) {
  return static_cast<sm_flag_t>(static_cast<unsigned>(lhs) |
                                static_cast<unsigned>(rhs));
}

/**
 * @brief Overloaded bitwise AND operator.
 *
 * @param lhs Left-hand side.
 * @param rhs Right-hand side.
 * \return sm_flag_t
 */
constexpr sm_flag_t operator&(sm_flag_t lhs, sm_flag_t rhs) {
  return static_cast<sm_flag_t>(static_cast<unsigned>(lhs) &
                                static_cast<unsigned>(rhs));
}

}  // namespace launch_box

}  // namespace gcuda
}  // namespace gunrock
