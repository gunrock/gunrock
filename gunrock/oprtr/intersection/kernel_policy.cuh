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
 * @brief Kernel configuration policy for Intersection Kernel
 */

#pragma once
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_details.cuh>
/*
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/app/problem_base.cuh>
*/
namespace gunrock {
namespace oprtr {
namespace intersection {

/**
 * @brief Kernel configuration policy for intersection kernels.
 *
 * Parameterizations of this type encapsulate our kernel-tuning parameters
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by
parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 *
 * @tparam _ProblemData                 Problem data type.
 * @tparam _CUDA_ARCH                   CUDA SM architecture to generate code
for.
<<<<<<< HEAD
=======
 * @tparam _INSTRUMENT                  Whether or not we want instrumentation
logic generated
>>>>>>> dev-intersection-op
 * @tparam _MIN_CTA_OCCUPANCY           Lower bound on number of CTAs to have
resident per SM (influences per-CTA smem cache sizes and register
allocation/spills).
 * @tparam _LOG_THREADS                 Number of threads per CTA (log).
 * @tparam _LOG_BLOCKS                  Number of blocks per grid (log).
 * @tparam _NL_SIZE_THRESHOLD           Threshold of neighbor list size when
doing intersection operation.
 */
template <OprtrFlag _FLAG,
          // typename _VertexT,      // Data types
          typename _InKeyT, typename _OutKeyT, typename _SizeT,
          typename _ValueT, typename _VertexT, typename _InterOpT,
          // Machine parameters
          // Tunable parameters
          size_t _MIN_CTA_OCCUPANCY, size_t _LOG_THREADS, size_t _LOG_BLOCKS,
          size_t _NL_SIZE_THRESHOLD>

struct KernelPolicy {
  //---------------------------------------------------------------------
  // Constants and typedefs
  //---------------------------------------------------------------------
  static const OprtrFlag FLAG = _FLAG;

  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  typedef _VertexT VertexT;
  typedef _InterOpT InterOpT;

  enum {

    MIN_CTA_OCCUPANCY = _MIN_CTA_OCCUPANCY,
    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
    NL_SIZE_THRESHOLD = 1 << _NL_SIZE_THRESHOLD,
  };

  /**
   * @brief Shared memory storage type for the CTA
   */
  struct SmemStorage {
    enum {
      MAX_SCRATCH_BYTES_PER_CTA = GR_SMEM_BYTES(CUDA_ARCH) / MIN_CTA_OCCUPANCY,

      SCRATCH_ELEMENT_SIZE = sizeof(SizeT),

      // for storing partition indices
      SCRATCH_ELEMENTS = THREADS + 1,
    };

    // Scratch elements
    struct {
      SizeT s_partition_idx[SCRATCH_ELEMENTS];  // stores block-wise
                                                // intersection counts
    };
  };

  enum {
    THREAD_OCCUPANCY = GR_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
    SMEM_OCCUPANCY = GR_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
    CTA_OCCUPANCY = GR_MIN(_MIN_CTA_OCCUPANCY,
                           GR_MIN(GR_SM_CTAS(CUDA_ARCH),
                                  GR_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),
    VALID = (CTA_OCCUPANCY > 0),
  };
};

}  // namespace intersection
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
