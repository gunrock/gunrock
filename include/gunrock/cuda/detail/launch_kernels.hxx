/**
 * @file launch_kernels.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-23
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/cuda/global.hxx>

namespace gunrock {
namespace gcuda {
namespace kernels {
namespace detail {
template <unsigned int threads_per_block,
          unsigned int items_per_thread,
          typename func_t,
          typename... args_t>
__global__ __launch_bounds__(threads_per_block,
                             items_per_thread)  // strict launch bounds
    void blocked_kernel(func_t f, const std::size_t bound, args_t... args) {
  const int stride = gcuda::block::size::x() * gcuda::grid::size::x();
  for (int i = gcuda::thread::global::id::x();  // global id
       i < bound;                               // bound check
       i += (stride * items_per_thread)         // offset
  ) {
#pragma unroll items_per_thread
    for (int j = 0; j < items_per_thread; ++j) {
      // Simple blocking per thread (unrolled items_per_thread_t)
      if ((i + (stride * j)) < bound)
        f(i + (stride * j), gcuda::block::id::x(), args...);
    }
  }
}

template <unsigned int threads_per_block, typename func_t, typename... args_t>
__global__ __launch_bounds__(threads_per_block, 1)  // strict launch bounds
    void strided_kernel(func_t f, const std::size_t bound, args_t... args) {
  const int stride = gcuda::block::size::x() * gcuda::grid::size::x();
  for (int i = gcuda::thread::global::id::x();  // global id
       i < bound;                               // bound check
       i += stride                              // offset
  ) {
    f(i, gcuda::block::id::x(), args...);
  }
}
}  // namespace detail
}  // namespace kernels
}  // namespace gcuda
}  // namespace gunrock