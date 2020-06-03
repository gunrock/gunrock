/**
 * @file device_properties.hxx
 *
 * @brief
 *
 *
 */

#pragma once
#include <gunrock/util/meta.hxx>

/**
 *  @see externals/cuda-api-wrappers/src/cuda/api/device_properties.hpp
 *     cuda::device::compute_architecture_t
 */
#include <cuda/api/device_properties.hpp>

namespace gunrock {
namespace util {

typedef cuda::device::compute_architecture_t architecture_t;

/**
 * @namespace properties
 * CUDA device properties namespace. Uses cuda-api-wrappers for
 * architecture specific values.
 */
namespace properties {

inline constexpr unsigned
shared_memory_banks()
{
  return 1 << 5; // 32 memory banks per SM
}

inline constexpr unsigned
shared_memory_bank_stride()
{
  return 1 << 2; // 4 byte words
}

inline constexpr unsigned
maximum_threads_per_warp()
{
  return 1 << 5; // 32 threads per warp
}

} // namespace properties

} // namespace util
} // namespace gunrock