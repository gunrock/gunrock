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
template<size_t x_, size_t y_ = 1, size_t z_ = 1>
struct dim3_t {
  enum : size_t { x = x_, y = y_, z = z_, size = x_ * y_ * z_ };
  static constexpr dim3 get_dim3() { return dim3(x, y, z); }
};

/**
 * @brief Struct holding kernel parameters will be passed in upon launch
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 */
template<typename block_dimensions_,
         typename grid_dimensions_,
         size_t shared_memory_bytes_ = 0>
struct launch_params_t {
  typedef block_dimensions_ block_dimensions;
  typedef grid_dimensions_ grid_dimensions;
  enum : size_t { shared_memory_bytes = shared_memory_bytes_ };
};

typedef unsigned sm_version_t;

/**
 * @brief Kernel parameters for a specific SM version
 * @tparam sm_version_ Combined major and minor compute capability version
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 */
template<sm_version_t sm_version_,
         typename block_dimensions_,
         typename grid_dimensions_,
         size_t shared_memory_bytes_ = 0>
struct sm_t : launch_params_t<block_dimensions_,
                              grid_dimensions_,
                              shared_memory_bytes_> {
  enum : sm_version_t {sm_version = sm_version_};
  static constexpr compute_capability_t get_compute_capability() {
    return make_compute_capability(sm_version);
  }
};

/**
  * @brief Kernel launch parmeters to fall back onto if the current device's
  * SM version isn't found
  * @tparam block_dimensions_ Block dimensions to launch with
  * @tparam grid_dimensions_ Grid dimensions to launch with
  * @tparam shared_memory_bytes_ Amount of shared memory to allocate
  */
template<typename block_dimensions_,
         typename grid_dimensions_,
         size_t shared_memory_bytes_ = 0>
struct fallback_t : sm_t<0,
                         block_dimensions_,
                         grid_dimensions_,
                         shared_memory_bytes_> {};

// Define named sm_t structs for each SM version
#define SM_LAUNCH_PARAMS(ver) \
template<typename block_dimensions_,         \
         typename grid_dimensions_,          \
         size_t shared_memory_bytes_ = 0>    \
using sm_##ver##_t = sm_t<ver,               \
                          block_dimensions_, \
                          grid_dimensions_,  \
                          shared_memory_bytes_>;

// Add Hopper when the SM version number becomes known (presumably 90)
SM_LAUNCH_PARAMS(86)
SM_LAUNCH_PARAMS(80)
SM_LAUNCH_PARAMS(75)
SM_LAUNCH_PARAMS(72)
SM_LAUNCH_PARAMS(70)
SM_LAUNCH_PARAMS(62)
SM_LAUNCH_PARAMS(61)
SM_LAUNCH_PARAMS(60)
SM_LAUNCH_PARAMS(53)
SM_LAUNCH_PARAMS(52)
SM_LAUNCH_PARAMS(50)
SM_LAUNCH_PARAMS(37)
SM_LAUNCH_PARAMS(35)
SM_LAUNCH_PARAMS(30)

#undef SM_LAUNCH_PARAMS

template<typename... sm_lp_v>
struct device_launch_params_t;

// First to second to last sm_t template parameters
template<typename sm_lp_t, typename... sm_lp_v>
struct device_launch_params_t<sm_lp_t, sm_lp_v...> :
std::conditional_t<sm_lp_t::sm_version == 0,
                   device_launch_params_t<sm_lp_v..., sm_lp_t>,  // Move fallback_t to end
                   std::conditional_t<sm_lp_t::sm_version == SM_TARGET,  // Otherwise check sm_lp_t for device's SM version
                                      sm_lp_t,
                                      device_launch_params_t<sm_lp_v...>>> {};

// "false", but dependent on a template parameter so the compiler can't
// optimize it for static_assert()
template<typename T>
struct always_false {
    enum { value = false };
};

// Raises static (compile-time) assert when template is instantiated
template<typename T>
struct raise_not_found_error_t {
  static_assert(always_false<T>::value,
                "Launch box could not find valid launch parameters");
};

// Last sm_t template parameter
template<typename sm_lp_t>
struct device_launch_params_t<sm_lp_t> :
std::conditional_t<
  sm_lp_t::sm_version == SM_TARGET || sm_lp_t::sm_version == 0,
  sm_lp_t,
  raise_not_found_error_t<void>  // Raises a compiler error
> {};

/**
 * @brief Collection of kernel launch parameters for multiple architectures
 * @tparam sm_lp_v... Pack of sm_t types for each desired arch
 */
template<typename... sm_lp_v>
struct launch_box_t : device_launch_params_t<sm_lp_v...> {};

/**
 * @brief Calculator for ratio of active to maximum warps per multiprocessor
 * @tparam launch_box_t Launch box for the corresponding kernel
 * @param kernel CUDA kernel for which to calculate the occupancy
 * \return float
 */
template<typename launch_box_t, typename func_t>
inline float occupancy(func_t kernel) {
  int max_active_blocks;
  int block_size = launch_box_t::block_dimensions::size;
  int device;
  cudaDeviceProp props;

  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  error::error_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks,
    kernel,
    block_size,
    (size_t)0
  );
  error::throw_if_exception(status);
  float occupancy = (max_active_blocks * block_size / props.warpSize) /
                    (float)(props.maxThreadsPerMultiProcessor /
                            props.warpSize);
  return occupancy;
}

}  // namespace launch_box

}  // namespace gunrock
}  // namespace cuda
