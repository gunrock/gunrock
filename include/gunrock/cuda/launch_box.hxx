/**
 * @file device_properties.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-11-09
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/error.hxx>

#include <type_traits>

#ifndef SM_TARGET
#define SM_TARGET 0
#endif

namespace gunrock {
namespace cuda {

namespace launch_box {

/**
 * @brief CUDA dim3 template representation, since dim3 cannot be used as a
 * template argument
 * @tparam x_ Dimension in the X direction
 * @tparam y_ Dimension in the Y direction
 * @tparam z_ Dimension in the Z direction
 */
template <unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1>
struct dim3_t {
  enum : unsigned int { x = x_, y = y_, z = z_, size = x_ * y_ * z_ };
  static constexpr dim3 get_dim3() { return dim3(x, y, z); }
};

enum sm_flag_t : unsigned {
  fallback = 0,
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

// Macro for the flag of the current device's SM version
#define SM_TARGET_FLAG _SM_FLAG_WRAPPER(SM_TARGET)
// "ver" will be expanded before the call to _SM_FLAG
#define _SM_FLAG_WRAPPER(ver) _SM_FLAG(ver)
#define _SM_FLAG(ver) sm_##ver

/**
 * @brief Overloaded bitwise OR operator
 * @param lhs Left-hand side
 * @param rhs Right-hand side
 * \return sm_flag_t
 */
constexpr sm_flag_t operator|(sm_flag_t lhs, sm_flag_t rhs) {
  return static_cast<sm_flag_t>(static_cast<unsigned>(lhs) |
                                static_cast<unsigned>(rhs));
}

/**
 * @brief Overloaded bitwise AND operator
 * @param lhs Left-hand side
 * @param rhs Right-hand side
 * \return sm_flag_t
 */
constexpr sm_flag_t operator&(sm_flag_t lhs, sm_flag_t rhs) {
  return static_cast<sm_flag_t>(static_cast<unsigned>(lhs) &
                                static_cast<unsigned>(rhs));
}

/**
 * @brief Kernel launch parameters for a specific SM version
 * @tparam sm_flags_ Bitwise flags indicating SM versions (sm_flag_t enum)
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 */
template <sm_flag_t sm_flags_,
          typename block_dimensions_,
          typename grid_dimensions_,
          size_t shared_memory_bytes_ = 0>
struct launch_params_t {
  typedef block_dimensions_ block_dimensions;
  typedef grid_dimensions_ grid_dimensions;
  enum : size_t { shared_memory_bytes = shared_memory_bytes_ };
  enum : unsigned { sm_flags = sm_flags_ };
};

template <typename... lp_v>
struct device_launch_params_t;

/**
 * @brief Struct that inherits the launch parameters of the current device
 * @tparam lp_t Launch parameters to check for match with current device
 * @tparam lp_v Pack of launch parameters to pass down recursively
 */
template <typename lp_t, typename... lp_v>
struct device_launch_params_t<lp_t, lp_v...>
    : std::conditional_t<
          lp_t::sm_flags == fallback,
          device_launch_params_t<lp_v..., lp_t>,  // Move fallback_t to end
          std::conditional_t<
              (bool)(lp_t::sm_flags& SM_TARGET_FLAG),  // Otherwise check lp_t
                                                       // for device's SM
                                                       // version
              lp_t,
              device_launch_params_t<lp_v...>>> {};

/**
 * @brief False value dependent on template param so compiler can't optimize
 * @tparam T Arbitrary type
 */
template <typename T>
struct always_false {
  enum { value = false };
};

/**
 * @brief Raises static assert when template is instantiated
 * @tparam T Arbitrary type
 */
template <typename T>
struct raise_not_found_error_t {
  static_assert(always_false<T>::value,
                "Launch box could not find valid launch parameters");
};

/**
 * @brief Struct that inherits the launch parameters of the current device
 * @tparam lp_t Launch parameters to check for match with current device or
 * fallback
 */
template <typename lp_t>
struct device_launch_params_t<lp_t>
    : std::conditional_t<
          (bool)(lp_t::sm_flags& SM_TARGET_FLAG) || lp_t::sm_flags == fallback,
          lp_t,
          raise_not_found_error_t<void>  // Raises a compiler error
          > {};

/**
 * @brief Collection of kernel launch parameters for multiple architectures
 * @tparam lp_v... Pack of launch_params_t types for each desired arch
 */
template <typename... lp_v>
struct launch_box_t : device_launch_params_t<lp_v...> {};

/**
 * @brief Calculator for ratio of active to maximum warps per multiprocessor
 * @tparam launch_box_t Launch box for the corresponding kernel
 * @param kernel CUDA kernel for which to calculate the occupancy
 * \return float
 */
template <typename launch_box_t, typename func_t>
inline float occupancy(func_t kernel) {
  int max_active_blocks;
  int block_size = launch_box_t::block_dimensions::size;
  int device;
  cudaDeviceProp props;

  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  error::error_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, kernel, block_size, (size_t)0);
  error::throw_if_exception(status);
  float occupancy = (max_active_blocks * block_size / props.warpSize) /
                    (float)(props.maxThreadsPerMultiProcessor / props.warpSize);
  return occupancy;
}

}  // namespace launch_box

}  // namespace cuda
}  // namespace gunrock
