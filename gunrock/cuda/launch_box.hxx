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

#include <type_traits>

namespace gunrock {
namespace cuda {

/**
 * @brief Struct holding kernel parameters will be passed in upon launch
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 *
 * @todo dimensions should be dim3 instead of unsigned int
 */
template<
  unsigned int block_dimensions_,
  unsigned int grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
struct launch_params_t {
  enum : unsigned int {
    block_dimensions = block_dimensions_,
    grid_dimensions = grid_dimensions_,
    shared_memory_bytes = shared_memory_bytes_
  };
};

#define TEST_SM 75  // Temporary until we figure out how to get cabability combined

/**
 * @brief Struct holding kernel parameters for a specific SM version
 * @tparam combined_ver_ Combined major and minor compute capability version
 * @tparam block_dimensions_ Block dimensions to launch with
 * @tparam grid_dimensions_ Grid dimensions to launch with
 * @tparam shared_memory_bytes_ Amount of shared memory to allocate
 */
template<
  unsigned int combined_ver_,
  unsigned int block_dimensions_,
  unsigned int grid_dimensions_,
  unsigned int shared_memory_bytes_ = 0
>
struct sm_launch_params_t : launch_params_t<
                              block_dimensions_,
                              grid_dimensions_,
                              shared_memory_bytes_
                            > {
  enum : unsigned int {combined_ver = combined_ver_};
};

// Easier declaration inside launch box template
template<
  unsigned int combined_ver_,
  unsigned int block_dimensions_,
  unsigned int grid_dimensions_,
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
  unsigned int block_dimensions_,                \
  unsigned int grid_dimensions_,                 \
  unsigned int shared_memory_bytes_ = 0          \
>                                                \
using sm_##combined##_t = sm_launch_params_t<    \
                            combined,            \
                            block_dimensions_,   \
                            grid_dimensions_,    \
                            shared_memory_bytes_ \
                          >;

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

template<typename sm_lp_t, typename... sm_lp_v>
struct device_launch_params_t<sm_lp_t, sm_lp_v...> :
std::conditional_t<
  sm_lp_t::combined_ver == TEST_SM,
  sm_lp_t,
  device_launch_params_t<sm_lp_v...>
> {};

template<typename sm_lp_t>
struct device_launch_params_t<sm_lp_t> :
std::enable_if_t<
  sm_lp_t::combined_ver == TEST_SM,
  sm_lp_t
> {};

/**
 * @brief Collection of kernel launch parameters for multiple architectures
 * @tparam sm_lp_v... Pack of sm_launch_params_t types for each desired arch
 */
template<typename... sm_lp_v>
struct launch_box_t : device_launch_params_t<sm_lp_v...> {
  // Some methods to make it easy to access launch_params
};

}  // namespace gunrock
}  // namespace cuda
