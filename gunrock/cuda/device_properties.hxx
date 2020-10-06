/**
 * @file device_properties.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

namespace gunrock {
namespace cuda {

/**
 * @namespace properties
 * C++ based CUDA device properties.
 */
namespace properties {

inline constexpr unsigned shared_memory_banks() {
  return 1 << 5;  // 32 memory banks per SM
}

inline constexpr unsigned shared_memory_bank_stride() {
  return 1 << 2;  // 4 byte words
}

inline constexpr unsigned maximum_threads_per_warp() {
  return 1 << 5;  // 32 threads per warp
}

}  // namespace properties

}  // namespace cuda
}  // namespace gunrock