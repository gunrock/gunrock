/**
 * @file sm.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
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
#if __HIP_PLATFORM_NVIDIA__
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
  sm_86 = 1 << 13,
  sm_87 = 1 << 14,
  sm_89 = 1 << 15,
  sm_90 = 1 << 16,
#else
  sm_gfx700 = 1 << 0,
  sm_gfx701 = 1 << 1,
  sm_gfx801 = 1 << 2,
  sm_gfx802 = 1 << 3,
  sm_gfx803 = 1 << 4,
  sm_gfx900 = 1 << 5,
  sm_gfx902 = 1 << 6,
  sm_gfx904 = 1 << 7,
  sm_gfx906 = 1 << 8,
  sm_gfx908 = 1 << 9,
  sm_gfx90a = 1 << 10,
  sm_gfx90c = 1 << 11,
  sm_gfx1010 = 1 << 12,
  sm_gfx1011 = 1 << 13,
  sm_gfx1012 = 1 << 14,
  sm_gfx1013 = 1 << 15,
  sm_gfx1030 = 1 << 16,
  sm_gfx1031 = 1 << 17,
  sm_gfx1032 = 1 << 18,
  sm_gfx1033 = 1 << 19,
  sm_gfx1034 = 1 << 20,
  sm_gfx1035 = 1 << 21,
#endif
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
