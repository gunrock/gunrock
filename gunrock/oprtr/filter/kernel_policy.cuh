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
 * @brief Kernel configuration policy for Filter Kernels
 */

#pragma once

// #include <gunrock/oprtr/cull_filter/kernel_policy.cuh>
#include <gunrock/oprtr/simplified_filter/kernel_policy.cuh>
#include <gunrock/oprtr/simplified2_filter/kernel_policy.cuh>
#include <gunrock/oprtr/compacted_cull_filter/kernel_policy.cuh>
// #include <gunrock/oprtr/bypass_filter/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace filter {

enum MODE { CULL, SIMPLIFIED, SIMPLIFIED2, COMPACTED_CULL, BY_PASS };

/**
 * @brief Kernel configuration policy for filter kernels.
 *
 * Parameterizations of this type encapsulate our kernel-tuning parameters
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by
 * parameterizing them with different performance-tuned parameterizations of
 * this type.  By incorporating this type into the kernel code itself, we guide
 * the compiler in expanding/unrolling the kernel code for specific
 * architectures and problem types.
 *
 * @tparam _ProblemData                 Problem data type.
 * @tparam _CUDA_ARCH                   CUDA SM architecture to generate code
 * for.
 * @tparam _INSTRUMENT                  Whether or not we want instrumentation
 * logic generated
 * @tparam _SATURATION_QUIT             If positive, signal that we're done with
 * two-phase iterations if frontier size drops below (SATURATION_QUIT *
 * grid_size).
 * @tparam _DEQUEUE_PROBLEM_SIZE        Whether we obtain problem size from
 * device-side queue counters (true), or use the formal parameter (false).
 * @tparam _MIN_CTA_OCCUPANCY           Lower bound on number of CTAs to have
 * resident per SM (influences per-CTA smem cache sizes and register
 * allocation/spills).
 * @tparam _LOG_THREADS                 Number of threads per CTA (log).
 * @tparam _LOG_LOAD_VEC_SIZE           Number of incoming frontier vertex-ids
 * to dequeue in a single load (log).
 * @tparam _LOG_LOADS_PER_TILE          Number of such loads that constitute a
 * tile of incoming frontier vertex-ids (log)
 * @tparam _LOG_RAKING_THREADS          Number of raking threads to use for
 * prefix sum (log), range [5, LOG_THREADS]
 * @tparam _END_BITMASK_CULL,           Iteration after which to skip bitmask
 * filtering (0 to never perform bitmask filtering, -1 to always perform bitmask
 * filtering)
 * @tparam _LOG_SCHEDULE_GRANULARITY    The scheduling granularity of incoming
 * frontier tiles (for even-share work distribution only) (log)
 */
template <typename _Problem,

          // Machine parameters
          int _CUDA_ARCH,
          // bool _INSTRUMENT,
          // Behavioral control parameters
          int _SATURATION_QUIT, bool _DEQUEUE_PROBLEM_SIZE,

          // Tunable parameters
          int _MAX_CTA_OCCUPANCY, int _LOG_THREADS, int _LOG_LOAD_VEC_SIZE,
          int _LOG_LOADS_PER_TILE, int _LOG_RAKING_THREADS,
          int _END_BITMASK_CULL, int _LOG_SCHEDULE_GRANULARITY,
          MODE _FILTER_MODE = CULL>
struct KernelPolicy {
  static const MODE FILTER_MODE = _FILTER_MODE;
  static const int CUDA_ARCH = _CUDA_ARCH;

  typedef gunrock::oprtr::cull_filter::KernelPolicy<
      _Problem, _CUDA_ARCH, _SATURATION_QUIT, _DEQUEUE_PROBLEM_SIZE,
      _MAX_CTA_OCCUPANCY, _LOG_THREADS, _LOG_LOAD_VEC_SIZE, _LOG_LOADS_PER_TILE,
      _LOG_RAKING_THREADS, _END_BITMASK_CULL, _LOG_SCHEDULE_GRANULARITY,
      _FILTER_MODE>
      CULL_FILTER;

  typedef gunrock::oprtr::simplified_filter::KernelPolicy<
      _Problem, _CUDA_ARCH, _SATURATION_QUIT, _DEQUEUE_PROBLEM_SIZE,
      _MAX_CTA_OCCUPANCY, _LOG_THREADS, _LOG_LOAD_VEC_SIZE, _LOG_LOADS_PER_TILE,
      _LOG_RAKING_THREADS, _END_BITMASK_CULL, _LOG_SCHEDULE_GRANULARITY,
      _FILTER_MODE>
      SIMPLIFIED_FILTER;

  typedef gunrock::oprtr::simplified2_filter::KernelPolicy<
      _Problem, _CUDA_ARCH, _SATURATION_QUIT, _DEQUEUE_PROBLEM_SIZE,
      _MAX_CTA_OCCUPANCY, _LOG_THREADS, _LOG_LOAD_VEC_SIZE, _LOG_LOADS_PER_TILE,
      _LOG_RAKING_THREADS, _END_BITMASK_CULL, _LOG_SCHEDULE_GRANULARITY,
      _FILTER_MODE>
      SIMPLIFIED2_FILTER;

  typedef gunrock::oprtr::compacted_cull_filter::KernelPolicy<
      _Problem, _CUDA_ARCH, _MAX_CTA_OCCUPANCY, _LOG_THREADS,
      _LOG_LOAD_VEC_SIZE + _LOG_LOADS_PER_TILE, _FILTER_MODE>
      COMPACTED_CULL_FILTER;

  typedef gunrock::oprtr::cull_filter::KernelPolicy<
      _Problem, _CUDA_ARCH, _SATURATION_QUIT, _DEQUEUE_PROBLEM_SIZE,
      _MAX_CTA_OCCUPANCY, _LOG_THREADS, _LOG_LOAD_VEC_SIZE, _LOG_LOADS_PER_TILE,
      _LOG_RAKING_THREADS, _END_BITMASK_CULL, _LOG_SCHEDULE_GRANULARITY,
      _FILTER_MODE>
      BYPASS_FILTER;

  static const int CTA_OCCUPANCY =
      (FILTER_MODE == CULL)
          ? CULL_FILTER ::CTA_OCCUPANCY
          : ((FILTER_MODE == SIMPLIFIED)
                 ? SIMPLIFIED_FILTER ::CTA_OCCUPANCY
                 : ((FILTER_MODE == COMPACTED_CULL)
                        ? COMPACTED_CULL_FILTER::CTA_OCCUPANCY
                        : ((FILTER_MODE == SIMPLIFIED2)
                               ? SIMPLIFIED2_FILTER ::CTA_OCCUPANCY
                               : ((FILTER_MODE == BY_PASS)
                                      ? BYPASS_FILTER ::CTA_OCCUPANCY
                                      : 0))));

  static const int THREADS =
      (FILTER_MODE == CULL)
          ? CULL_FILTER ::THREADS
          : ((FILTER_MODE == SIMPLIFIED)
                 ? SIMPLIFIED_FILTER ::THREADS
                 : ((FILTER_MODE == COMPACTED_CULL)
                        ? COMPACTED_CULL_FILTER::THREADS
                        : ((FILTER_MODE == SIMPLIFIED2)
                               ? SIMPLIFIED2_FILTER ::THREADS
                               : ((FILTER_MODE == BY_PASS)
                                      ? BYPASS_FILTER ::THREADS
                                      : 0))));
};

}  // namespace filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
