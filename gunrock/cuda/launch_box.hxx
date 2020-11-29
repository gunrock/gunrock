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

#include <gunrock/error.hxx>

#include <type_traits>

#ifndef SM_TARGET
  #define SM_TARGET 0
#endif

namespace gunrock {
namespace cuda {

/**
 * @brief CUDA dim3 template representation, since dim3 cannot be used as a
 * template argument
 * @tparam x_ Dimension in the X direction
 * @tparam y_ Dimension in the Y direction
 * @tparam z_ Dimension in the Z direction
 */
template<unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1>
struct dim3_t {
  enum : unsigned int { x = x_, y = y_, z = z_, size = x_ * y_ * z_ };
  static dim3 get_dim3() { return dim3(x, y, z); }
};

/**
 * @brief Struct holding kernel parameters will be passed in upon launch
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 *
 * @todo dimensions should be dim3 instead of unsigned int
 */
template<
  typename block_dimensions_,
  typename grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
struct launch_params_t {
  typedef block_dimensions_ block_dimensions;
  typedef grid_dimensions_ grid_dimensions;
  enum : unsigned int { shared_memory_bytes = shared_memory_bytes_ };
};

/**
 * @brief Struct holding kernel parameters for a specific SM version
 * @tparam combined_ver_ Combined major and minor compute capability version
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 */
template<
  unsigned int combined_ver_,
  typename block_dimensions_,
  typename grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
struct sm_launch_params_t : launch_params_t<
                              block_dimensions_,
                              grid_dimensions_,
                              shared_memory_bytes_
                            > {
  enum : unsigned int {combined_ver = combined_ver_};
};

template<
  typename block_dimensions_,
  typename grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
struct fallback_launch_params_t : sm_launch_params_t<
                                    0,
                                    block_dimensions_,
                                    grid_dimensions_,
                                    shared_memory_bytes_
                                  > {};

// Easier declaration inside launch box template
template<
  typename block_dimensions_,
  typename grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
using fallback_t = fallback_launch_params_t<
                     block_dimensions_,
                     grid_dimensions_,
                     shared_memory_bytes_
                   >;

// Easier declaration inside launch box template
template<
  unsigned int combined_ver_,
  typename block_dimensions_,
  typename grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
using sm_t = sm_launch_params_t<
               combined_ver_,
               block_dimensions_,
               grid_dimensions_,
               shared_memory_bytes_
             >;

// Define named sm_launch_params_t structs for each SM version
#define SM_LAUNCH_PARAMS(combined) \
template<                                        \
  typename block_dimensions_,                    \
  typename grid_dimensions_,                     \
  unsigned int shared_memory_bytes_ = 0          \
>                                                \
using sm_##combined##_t = sm_launch_params_t<    \
                            combined,            \
                            block_dimensions_,   \
                            grid_dimensions_,    \
                            shared_memory_bytes_ \
                          >;

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

// 1st to (N-1)th sm_launch_param_t template parameters
template<typename sm_lp_t, typename... sm_lp_v>
struct device_launch_params_t<sm_lp_t, sm_lp_v...> :
std::conditional_t<
  sm_lp_t::combined_ver == 0,
  device_launch_params_t<sm_lp_v..., sm_lp_t>,  // If found, move fallback_launch_params_t to end of template parameter pack
  std::conditional_t<                           // Otherwise check sm_lp_t for device's SM version
    sm_lp_t::combined_ver == SM_TARGET,
    sm_lp_t,
    device_launch_params_t<sm_lp_v...>
  >
> {};

//////////////// https://stackoverflow.com/a/3926854
// "false" but dependent on a template parameter so the compiler can't optimize it for static_assert()
template<typename T>
struct always_false {
    enum { value = false };
};

// Raises static (compile-time) assert when referenced
template<typename T>
struct raise_not_found_error_t {
  static_assert(always_false<T>::value, "launch_box_t could not find valid launch_params_t");
};
////////////////

// Nth sm_launch_param_t template parameter
template<typename sm_lp_t>
struct device_launch_params_t<sm_lp_t> :
std::conditional_t<
  sm_lp_t::combined_ver == SM_TARGET || sm_lp_t::combined_ver == 0,
  sm_lp_t,
  raise_not_found_error_t<void>  // Raises a compiler error
> {};

/**
 * @brief Collection of kernel launch parameters for multiple architectures
 * @tparam sm_lp_v... Pack of sm_launch_params_t types for each desired arch
 */
template<typename... sm_lp_v>
struct launch_box_t : device_launch_params_t<sm_lp_v...> {
  // Some methods to make it easy to access launch_params
};

/**
 * @brief Calculator for ratio of active to maximum warps per multiprocessor
 * @tparam launch_box_t Launch box for the corresponding kernel
 * @param kernel CUDA kernel for which to calculate the occupancy
 * \return float
 */
template<typename launch_box_t, typename func_t>
float occupancy(func_t kernel) {
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

}  // namespace gunrock
}  // namespace cuda
