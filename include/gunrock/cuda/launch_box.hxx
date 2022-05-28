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

#include <gunrock/error.hxx>

#include <gunrock/cuda/sm.hxx>
#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/cuda/context.hxx>

#include <gunrock/cuda/detail/launch_box.hxx>
#include <gunrock/cuda/detail/launch_kernels.hxx>

#include <tuple>
#include <type_traits>

#ifndef SM_TARGET
#define SM_TARGET 0
#endif

namespace gunrock {
namespace gcuda {
namespace launch_box {

struct dimensions_t {
  unsigned int x, y, z;

  __host__ __device__ constexpr dimensions_t(const unsigned int _x = 1,
                                             const unsigned int _y = 1,
                                             const unsigned int _z = 1)
      : x(_x), y(_y), z(_z) {}

  __host__ __device__ constexpr unsigned int size() const { return x * y * z; }

#ifdef _MSC_VER
  __host__ __device__ operator dim3(void) const { return uint3{x, y, z}; }
#else
  __host__ __device__ constexpr operator dim3(void) const {
    return uint3{x, y, z};
  }
#endif
};

/**
 * @brief CUDA dim3 template representation, since dim3 cannot be used as a
 * template argument.
 *
 * @tparam x_ (default = `1`) Dimension in the X direction.
 * @tparam y_ (default = `1`) Dimension in the Y direction.
 * @tparam z_ (default = `1`) Dimension in the Z direction.
 */
template <unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1>
struct dim3_t {
  enum : unsigned int { x = x_, y = y_, z = z_ };
  static constexpr unsigned int size() { return x * y * z; }
  static constexpr dimensions_t dimensions() { return {x, y, z}; }

  // Convertors must be non-static members.
  constexpr operator dimensions_t(void) { return {x, y, z}; }
};

/**
 * @brief Set of launch parameters for a CUDA kernel.
 *
 * @tparam sm_flags_ Bit flags for the SM architectures the launch parameters
 * correspond to.
 * @tparam block_dimensions_ A `dim3_t` type representing the block dimensions.
 * @tparam grid_dimensions_ A `dim3_t` type representing the grid dimensions.
 * @tparam items_per_thread_ (default = `1`) Number of items per thread.
 * @tparam shared_memory_bytes_ (default = `0`) Number of bytes of shared memory
 * to allocate.
 */
template <sm_flag_t sm_flags_,
          typename block_dimensions_,
          typename grid_dimensions_,
          std::size_t items_per_thread_ = 1,
          std::size_t shared_memory_bytes_ = 0>
struct launch_params_t : detail::launch_params_base_t<sm_flags_,
                                                      items_per_thread_,
                                                      shared_memory_bytes_> {
  typedef detail::
      launch_params_base_t<sm_flags_, items_per_thread_, shared_memory_bytes_>
          base_t;
  typedef block_dimensions_ block_dimensions_t;
  typedef grid_dimensions_ grid_dimensions_t;

  static constexpr dimensions_t block_dimensions =
      block_dimensions_t::dimensions();
  static constexpr dimensions_t grid_dimensions =
      grid_dimensions_t::dimensions();
};

/**
 * @brief Set of launch parameters for a CUDA kernel (with dynamic grid
 * dimensions).
 *
 * @tparam sm_flags_ Bit flags for the SM architectures the launch parameters
 * correspond to.
 * @tparam block_dimensions_ A `dim3_t` type representing the block dimensions.
 * @tparam items_per_thread_ (default = `1`) Number of items per thread.
 * @tparam shared_memory_bytes_ (default = `0`) Number of bytes of shared memory
 * to allocate.
 */
template <sm_flag_t sm_flags_,
          typename block_dimensions_,
          std::size_t items_per_thread_ = 1,
          std::size_t shared_memory_bytes_ = 0>
struct launch_params_dynamic_grid_t
    : detail::launch_params_base_t<sm_flags_,
                                   items_per_thread_,
                                   shared_memory_bytes_> {
  typedef detail::
      launch_params_base_t<sm_flags_, items_per_thread_, shared_memory_bytes_>
          base_t;
  typedef block_dimensions_ block_dimensions_t;

  static constexpr dimensions_t block_dimensions =
      block_dimensions_t::dimensions();

  dimensions_t grid_dimensions;

  void calculate_grid_dimensions_strided(std::size_t num_elements) {
    grid_dimensions = dimensions_t(
        (num_elements + block_dimensions.x - 1) / block_dimensions.x, 1, 1);
  }

  void calculate_grid_dimensions_blocked(std::size_t num_elements) {
    grid_dimensions = dimensions_t(
        (num_elements + (block_dimensions.x * base_t::items_per_thread) - 1) /
            (block_dimensions.x * base_t::items_per_thread),
        1, 1);
  }
};

/**
 * @brief Alias a selected a launch params type from valid options on
 * architecture being compiled for.
 *
 * @tparam lp_v Pack of `launch_params_t` types for each corresponding
 * architecture(s).
 */
template <typename... lp_v>
using select_launch_params_t = std::conditional_t<
    (std::tuple_size<detail::match_launch_params_t<lp_v...>>::value == 0),
    detail::raise_not_found_error_t<void>,  // Couldn't find arch params
    std::tuple_element_t<0, detail::match_launch_params_t<lp_v...>>>;

/**
 * @brief Collection of kernel launch parameters for different
 * architectures.
 *
 * @par Overview
 * A launch box is a collection of sets of CUDA kernel launch parameters
 * each corresponding to one or more SM architectures. At compile time, the
 * launch box's type resolves to the **first** launch parameters type
 * (derived from `launch_param_base_t`) that match the SM architecture that
 * Gunrock is being compiled for. If there isn't an explicit match, launch
 * parameters for any SM version can be specified at the end of the
 * parameter pack using the `fallback` enum for the `sm_flag_t` template
 * parameter (note that this will invalidate any launch parameter types
 * later in the parameter pack). In the case that there isn't a fallback and
 * the compiler can't find launch parameters for the architecture being
 * compiled for, a static assert will be raised. All launch parameters
 * *should* use the same struct template so there isn't any ambiguity as to
 * what the type the launch box is inheriting, though this isn't enforced.
 *
 * @par Example
 * The following code is an example of how to instantiate a launch box.
 *
 * \code
 * typedef launch_box_t<
 *     Type, SM Arch, Block Dimension, Grid Dimension, Items/Thread, Shared Mem
 *     launch_params_t<sm_86 | sm_80, dim3_t<16, 2, 2>, dim3_t<4, 1, 4>, 2>,
 *     launch_params_t<sm_75 | sm_70, dim3_t<32, 2, 4>, dim3_t<64, 8, 8>>,
 *     launch_params_t<sm_61 | sm_60, dim3_t<8, 4, 4>, dim3_t<32, 1, 4>, 2>,
 *     launch_params_t<sm_35, dim3_t<64>, dim3_t<64>, 16>,
 *     launch_params_t<fallback, dim3_t<16>, dim3_t<2>, 4>>
 *     launch_t;
 *
 * launch_t my_launch_box;
 * \endcode
 *
 * @tparam lp_v Pack of `launch_params_t` types for each corresponding
 * architecture(s).
 */
template <typename... lp_v>
struct launch_box_t : public select_launch_params_t<lp_v...> {
  typedef select_launch_params_t<lp_v...> params_t;

  // Empty constructor to avoid compiler warnings.
  launch_box_t(){};

  /**
   * @brief Launch a function with the given parameters on a simple strided
   * kernel.
   *
   * @par Overview
   * This function is a simple wrapper around the CUDA kernel launch, the
   * kernel expects a function f, which a signature f(int thread_id, int
   * block_id, ...). An example usage of this function is as follows:
   *
   * \code
   * std::size_t num_elements = 1024;
   * auto f = [=] __device__(int const& tid, int const& bid) {
   *  // Do something
   * };
   * using namespace gcuda::launch_box;
   * using launch_t =
   *  launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<128>>>;
   *
   * launch_t launch_box;
   * launch_box.calculate_grid_dimensions(num_elements);
   * launch_box.launch_strided(context, f, num_elements);
   * context.synchronize();
   * \endcode
   *
   * @tparam func_t Function type to launch.
   * @tparam args_t Pack of arguments to pass to the kernel.
   * @param context The device context to use.
   * @param f lambda, function or functor to be launched.
   * @param num_elements if thread id > num_elements, return.
   * @param args arguments to be passed to the function.
   */
  template <typename func_t, typename... args_t>
  void launch_strided(gcuda::standard_context_t& context,
                      func_t& f,
                      const std::size_t num_elements,
                      args_t&&... args) {
    params_t::calculate_grid_dimensions_strided(num_elements);
    using namespace kernels::detail;
    strided_kernel<params_t::block_dimensions_t::size()>
        <<<params_t::grid_dimensions, params_t::block_dimensions,
           params_t::shared_memory_bytes, context.stream()>>>(
            f, num_elements, std::forward<args_t>(args)...);
  }

  /**
   * @brief Launch a function with the given parameters on a simple strided and
   * blocked kernel. The blocking is defined using the
   * launch_box_t::items_per_thread template parameter.
   *
   * @par Overview
   * This function is a simple wrapper around the CUDA kernel launch, the
   * kernel expects a function f, which a signature f(int thread_id, int
   * block_id, ...). An example usage of this function is as follows:
   *
   * \code
   * std::size_t num_elements = 1024;
   * auto f = [=] __device__(int const& tid, int const& bid) {
   *  // Do something
   * };
   * using namespace gcuda::launch_box;
   * using launch_t =
   *  launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<128>>>;
   *
   * launch_t launch_box;
   * launch_box.calculate_grid_dimensions(num_elements);
   * launch_box.launch_blocked_strided(context, f, num_elements);
   * context.synchronize();
   * \endcode
   *
   * @tparam func_t Function type to launch.
   * @tparam args_t Pack of arguments to pass to the kernel.
   * @param context The device context to use.
   * @param f lambda, function or functor to be launched.
   * @param num_elements if thread id > num_elements, return.
   * @param args arguments to be passed to the function.
   */
  template <typename func_t, typename... args_t>
  void launch_blocked(gcuda::standard_context_t& context,
                      func_t& f,
                      const std::size_t num_elements,
                      args_t&&... args) {
    params_t::calculate_grid_dimensions_blocked(num_elements);
    using namespace kernels::detail;
    blocked_kernel<params_t::block_dimensions_t::size(),
                   params_t::items_per_thread>
        <<<params_t::grid_dimensions, params_t::block_dimensions,
           params_t::shared_memory_bytes, context.stream()>>>(
            f, num_elements, std::forward<args_t>(args)...);
  }

  template <typename func_t, typename... args_t>
  void launch_cooperative(gcuda::standard_context_t& context,
                          const func_t& f,
                          const std::size_t num_elements,
                          args_t&&... args) {
    params_t::calculate_grid_dimensions_strided(num_elements);
    constexpr const auto non_zero_num_params =
        sizeof...(args_t) == 0 ? 1 : sizeof...(args_t);
    void* argument_ptrs[non_zero_num_params];
    detail::for_each_argument_address(argument_ptrs,
                                      ::std::forward<args_t>(args)...);
    cudaLaunchCooperativeKernel<func_t>(
        &f, params_t::grid_dimensions, params_t::block_dimensions,
        argument_ptrs, params_t::shared_memory_bytes, context.stream());
  }

  /**
   * @brief Launch a kernel within the given launch box.
   *
   * @par Overview
   * This function is a reimplementation of `std::apply`, that allows
   * for launching cuda kernels with launch param members of the class
   * and a context argument. It follows the "possible implementation" of
   * `std::apply` in the C++ reference:
   * https://en.cppreference.com/w/cpp/utility/apply.
   *
   * @tparam func_t The type of the kernel function being passed in.
   * @tparam args_tuple_t The type of the tuple of arguments being
   * passed in.
   * @param f Kernel function to call.
   * @param args Tuple of arguments to be expanded as the arguments of
   * the kernel function.
   * @param context Reference to the context used to launch the kernel
   * (used for the context's stream).
   * \return void
   */
  template <typename func_t, typename... args_t>
  void launch(gcuda::standard_context_t& context,
              const func_t& f,
              args_t&&... args) {
    f<<<params_t::grid_dimensions, params_t::block_dimensions,
        params_t::shared_memory_bytes, context.stream()>>>(
        std::forward<args_t>(args)...);
  }
};  // struct launch_box_t

/**
 * @brief Calculator for ratio of active to maximum warps per multiprocessor.
 *
 * @tparam launch_box_t Launch box for the corresponding kernel.
 * @tparam func_t The type of the kernel function being passed in.
 * @param kernel CUDA kernel for which to calculate the occupancy.
 * \return float
 */
template <typename launch_box_t, typename func_t>
inline float occupancy(func_t kernel) {
  int max_active_blocks;
  int block_size = launch_box_t::block_dimensions_t::size();
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
}  // namespace gcuda
}  // namespace gunrock
