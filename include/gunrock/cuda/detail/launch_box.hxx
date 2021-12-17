/**
 * @file launch_box.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-16
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

namespace gunrock {
namespace cuda {

namespace launch_box {

/**
 * @brief Bit flag enum representing different SM architectures.
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

namespace detail {

/**
 * @brief Abstract base class for launch parameters.
 *
 * @tparam sm_flags_ Bitwise flags indicating SM versions (`sm_flag_t` enum).
 */
template <sm_flag_t sm_flags_>
struct launch_params_abc_t {
  enum : unsigned { sm_flags = sm_flags_ };

 protected:
  launch_params_abc_t();
};

/**
 * @brief False value dependent on template param so compiler can't optimize.
 *
 * @tparam T Arbitrary type.
 */
template <typename T>
struct always_false {
  enum { value = false };
};

/**
 * @brief Raises static assert when template is instantiated.
 *
 * @tparam T Arbitrary type.
 */
template <typename T>
struct raise_not_found_error_t {
  static_assert(always_false<T>::value,
                "Launch box could not find valid launch parameters");
};

// Macro for the flag of the current device's SM version
#define SM_TARGET_FLAG _SM_FLAG_WRAPPER(SM_TARGET)
// "ver" will be expanded before the call to _SM_FLAG
#define _SM_FLAG_WRAPPER(ver) _SM_FLAG(ver)
#define _SM_FLAG(ver) sm_##ver

/**
 * @brief Subsets a pack of launch parameters (children of
 * `launch_params_abc_t`), selecting the ones that match the architecture being
 * compiled for.
 *
 * @tparam lp_v Pack of `launch_params_t` types for each desired arch.
 */
template <typename... lp_v>
using match_launch_params_t = decltype(std::tuple_cat(
    std::declval<std::conditional_t<(bool)(lp_v::sm_flags& SM_TARGET_FLAG),
                                    std::tuple<lp_v>,
                                    std::tuple<>>>()...));

}  // namespace detail
}  // namespace launch_box

}  // namespace cuda
}  // namespace gunrock
