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
 * @brief Kernel configuration policy for Load balanced Edge Expansion Kernel
 */

#pragma once

namespace gunrock {
namespace oprtr {
namespace LB {

/**
 * @brief Kernel configuration policy for partitioned edge mapping kernels.
 *
 * Parameterizations of this type encapsulate our kernel-tuning parameters
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by
 * parameterizing them with different performance-tuned parameterizations of
 * this type.  By incorporating this type into the kernel code itself, we guide
 * the compiler in expanding/unrolling the kernel code for specific
 * architectures and problem types.
 *
 * @tparam _MIN_CTA_OCCUPANCY           Lower bound on number of CTAs to have
 * resident per SM (influences per-CTA smem cache sizes and register
 * allocation/spills).
 * @tparam _LOG_THREADS                 Number of threads per CTA (log).
 */
template <typename _VertexT,  // Data types
          typename _InKeyT, typename _SizeT, typename _ValueT,
          int _MAX_CTA_OCCUPANCY,  // Tunable parameters
          int _LOG_THREADS, int _LOG_BLOCKS, int _LIGHT_EDGE_THRESHOLD>
struct KernelPolicy {
  //---------------------------------------------------------------------
  // Constants and typedefs
  //---------------------------------------------------------------------

  typedef _VertexT VertexT;
  typedef _InKeyT InKeyT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;

  enum {

    // CUDA_ARCH                       = _CUDA_ARCH,
    // INSTRUMENT                      = _INSTRUMENT,

    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
    LIGHT_EDGE_THRESHOLD = _LIGHT_EDGE_THRESHOLD,
  };

  enum {
    // Amount of storage we can use for hashing scratch space under target
    // occupancy
    // MAX_SCRATCH_BYTES_PER_CTA       = (GR_SMEM_BYTES(CUDA_ARCH) /
    // _MIN_CTA_OCCUPANCY)
    //                                    - 128, // Fudge-factor to guarantee
    //                                    occupancy

    // SCRATCH_ELEMENT_SIZE            = sizeof(SizeT) * 2 + sizeof(VertexId) *
    // 2,

    SCRATCH_ELEMENTS =
        256,  //(THREADS > MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE) ?
              //MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE : THREADS,
  };

  /**
   * @brief Shared memory storage type for the CTA
   */
  struct SmemStorage {
    // Scratch elements
    struct {
      SizeT output_offset[SCRATCH_ELEMENTS];
      SizeT row_offset[SCRATCH_ELEMENTS];
      VertexT vertices[SCRATCH_ELEMENTS];
      InKeyT input_queue[SCRATCH_ELEMENTS];
    };
  };

  enum {
    THREAD_OCCUPANCY = GR_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
    SMEM_OCCUPANCY = GR_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
    CTA_OCCUPANCY = GR_MIN(_MAX_CTA_OCCUPANCY,
                           GR_MIN(GR_SM_CTAS(CUDA_ARCH),
                                  GR_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

    VALID = (CTA_OCCUPANCY > 0),
  };
};

}  // namespace LB
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
