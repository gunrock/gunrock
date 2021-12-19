/**
 * @file launch_box.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-11-09
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/cuda/detail/launch_box.hxx>
#include <gunrock/cuda/sm.hxx>
#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/cuda/context.hxx>
#include <gunrock/error.hxx>

#include <tuple>
#include <type_traits>

#ifndef SM_TARGET
#define SM_TARGET 0
#endif

namespace gunrock {
namespace cuda {

namespace launch_box {

/**
 * @brief CUDA dim3 template representation, since dim3 cannot be used as a
 * template argument.
 *
 * @tparam x_ Dimension in the X direction.
 * @tparam y_ (default = `1`) Dimension in the Y direction.
 * @tparam z_ (default = `1`) Dimension in the Z direction.
 */
template <unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1>
struct dim3_t {
  enum : unsigned int { x = x_, y = y_, z = z_, size = x_ * y_ * z_ };
  static constexpr dim3 get_dim3() { return dim3(x, y, z); }
};

/**
 * @brief Collection of kernel launch parameters for different architectures.
 *
 * @par Overview
 * A launch box is a collection of sets of CUDA kernel launch parameters each
 * corresponding to one or more SM architectures. At compile time, the launch
 * box's type resolves to the **first** launch parameters type (derived from
 * `launch_param_abc_t`) that match the SM architecture that Gunrock is being
 * compiled for. If there isn't an explicit match, launch parameters for any SM
 * version can be specified at the end of the parameter pack using the
 * `fallback` enum for the `sm_flag_t` template parameter (note that this will
 * invalidate any launch parameter types later in the parameter pack). In the
 * case that there isn't a fallback and the compiler can't find launch
 * parameters for the architecture being compiled for, a static assert will be
 * raised. All launch parameters *should* use the same struct template so there
 * isn't any ambiguity as to what the launch box's constructor is, though this
 * isn't enforced.
 *
 * @par Example
 * The following code is an example of how to instantiate a launch box.
 *
 * \code
 * typedef launch_box_t<
 *     launch_params_t<sm_86 | sm_80, dim3_t<16, 2, 2>, dim3_t<64, 1, 4>, 2>,
 *     launch_params_t<sm_75 | sm_70, dim3_t<32, 2, 4>, dim3_t<64, 8, 8>>,
 *     launch_params_t<sm_61 | sm_60, dim3_t<8, 4, 4>, dim3_t<32, 1, 4>, 2>,
 *     launch_params_t<sm_35, dim3_t<64>, dim3_t<64>, 16>,
 *     launch_params_t<fallback, dim3_t<16>, dim3_t<2>, 4>>
 *     launch_t;
 *
 * launch_t my_launch_box(context);
 * \endcode
 *
 * @tparam lp_v Pack of `launch_params_t` types for each corresponding
 * architecture(s).
 */
template <typename... lp_v>
using launch_box_t = std::conditional_t<
    (std::tuple_size<detail::match_launch_params_t<lp_v...>>::value == 0),
    detail::raise_not_found_error_t<void>,  // Couldn't find params for SM ver
    std::tuple_element_t<0, detail::match_launch_params_t<lp_v...>>>;

/**
 * @brief Set of launch parameters for a CUDA kernel.
 *
 * @tparam sm_flags_ Bit flags for the SM architectures the launch parameters
 * correspond to.
 * @tparam block_dimensions_ A `dim3_t` type representing the block dimensions.
 * @tparam grid_dimensions_ A `dim3_t` type representing the grid dimensions.
 * @tparam shared_memory_bytes_ Number of bytes of shared memory to allocate.
 */
template <sm_flag_t sm_flags_,
          typename block_dimensions_,
          typename grid_dimensions_,
          size_t shared_memory_bytes_ = 0>
struct launch_params_t : detail::launch_params_abc_t<sm_flags_> {
  typedef block_dimensions_ block_dimensions_t;
  typedef grid_dimensions_ grid_dimensions_t;
  enum : size_t { shared_memory_bytes = shared_memory_bytes_ };

  static constexpr dim3 block_dimensions = block_dimensions_t::get_dim3();
  static constexpr dim3 grid_dimensions = grid_dimensions_t::get_dim3();
  standard_context_t& context;

  launch_params_t(standard_context_t& context_) : context(context_) {}
};

/**
 * @brief Set of launch parameters for a CUDA kernel (with non-static block
 * dimensions).
 *
 * @tparam sm_flags_ Bit flags for the SM architectures the launch parameters
 * correspond to.
 * @tparam grid_dimensions_ A `dim3_t` type representing the grid dimensions.
 * @tparam shared_memory_bytes_ Number of bytes of shared memory to allocate.
 */
template <sm_flag_t sm_flags_,
          typename grid_dimensions_,
          size_t shared_memory_bytes_ = 0>
struct launch_params_dynamic_block_t : detail::launch_params_abc_t<sm_flags_> {
  typedef grid_dimensions_ grid_dimensions_t;
  enum : size_t { shared_memory_bytes = shared_memory_bytes_ };

  dim3 block_dimensions;
  static constexpr dim3 grid_dimensions = grid_dimensions_t::get_dim3();
  standard_context_t& context;

  launch_params_dynamic_block_t(dim3 block_dimensions_,
                                standard_context_t& context_)
      : block_dimensions(block_dimensions_), context(context_) {}
};

/**
 * @brief Set of launch parameters for a CUDA kernel (with non-static grid
 * dimensions).
 *
 * @tparam sm_flags_ Bit flags for the SM architectures the launch parameters
 * correspond to.
 * @tparam block_dimensions_ A `dim3_t` type representing the block dimensions.
 * @tparam shared_memory_bytes_ Number of bytes of shared memory to allocate.
 */
template <sm_flag_t sm_flags_,
          typename block_dimensions_,
          size_t shared_memory_bytes_ = 0>
struct launch_params_dynamic_grid_t : detail::launch_params_abc_t<sm_flags_> {
  typedef block_dimensions_ block_dimensions_t;
  enum : size_t { shared_memory_bytes = shared_memory_bytes_ };

  static constexpr dim3 block_dimensions = block_dimensions_t::get_dim3();
  dim3 grid_dimensions;
  standard_context_t& context;

  launch_params_dynamic_grid_t(dim3 grid_dimensions_,
                               standard_context_t& context_)
      : grid_dimensions(grid_dimensions_), context(context_) {}
};

/**
 * @brief Calculator for ratio of active to maximum warps per multiprocessor.
 *
 * @tparam launch_box_t Launch box for the corresponding kernel.
 * @param kernel CUDA kernel for which to calculate the occupancy.
 * \return float
 */
template <typename launch_box_t, typename func_t>
inline float occupancy(func_t kernel) {
  int max_active_blocks;
  int block_size = launch_box_t::block_dimensions_t::size;
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
