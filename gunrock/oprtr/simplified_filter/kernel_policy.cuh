// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel_policy.cuh
 *
 * @brief Kernel configuration policy for simplified filter
 */

#pragma once

// #include <gunrock/oprtr/cull_filter/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace simplified_filter {

template <typename _Problem,

          // Machine parameters
          int _CUDA_ARCH,
          // bool _INSTRUMENT,
          // Behavioral control parameters
          int _SATURATION_QUIT, bool _DEQUEUE_PROBLEM_SIZE,

          // Tunable parameters
          int _MIN_CTA_OCCUPANCY, int _LOG_THREADS, int _LOG_LOAD_VEC_SIZE,
          int _LOG_LOADS_PER_TILE, int _LOG_RAKING_THREADS,
          int _END_BITMASK_CULL, int _LOG_SCHEDULE_GRANULARITY, int _MODE>
struct KernelPolicy
    : public gunrock::oprtr::cull_filter::KernelPolicy<
          _Problem, _CUDA_ARCH, _SATURATION_QUIT, _DEQUEUE_PROBLEM_SIZE,
          _MIN_CTA_OCCUPANCY, _LOG_THREADS, _LOG_LOAD_VEC_SIZE,
          _LOG_LOADS_PER_TILE, _LOG_RAKING_THREADS, _END_BITMASK_CULL,
          _LOG_SCHEDULE_GRANULARITY, _MODE> {
  enum {
    MODE = _MODE,
  };
};

}  // namespace simplified_filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
