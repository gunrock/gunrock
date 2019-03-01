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
namespace simplified_advance {

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
 * @tparam _ProblemData                 Problem data type.
 * @tparam _CUDA_ARCH                   CUDA SM architecture to generate code
 * for.
 * @tparam _INSTRUMENT                  Whether or not we want instrumentation
 * logic generated
 * @tparam _MIN_CTA_OCCUPANCY           Lower bound on number of CTAs to have
 * resident per SM (influences per-CTA smem cache sizes and register
 * allocation/spills).
 * @tparam _LOG_THREADS                 Number of threads per CTA (log).
 */
template <typename _ProblemData,
          // Machine parameters
          int _CUDA_ARCH,
          // Tunable parameters
          int _MIN_CTA_OCCUPANCY, int _LOG_THREADS, int _LOG_BLOCKS,
          int _LIGHT_EDGE_THRESHOLD>

struct KernelPolicy {
  //---------------------------------------------------------------------
  // Constants and typedefs
  //---------------------------------------------------------------------

  typedef _ProblemData ProblemData;
  typedef typename ProblemData::VertexId VertexId;
  typedef typename ProblemData::SizeT SizeT;
  typedef typename ProblemData::Value Value;

  enum {

    CUDA_ARCH = _CUDA_ARCH,
    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
    LIGHT_EDGE_THRESHOLD = _LIGHT_EDGE_THRESHOLD,
  };

  enum {
    SCRATCH_ELEMENTS = 256,
  };

  /**
   * @brief Shared memory storage type for the CTA
   */
  struct SmemStorage {
    // Scratch elements
    struct {
      SizeT output_offset[SCRATCH_ELEMENTS];
      SizeT row_offset[SCRATCH_ELEMENTS];
      VertexId vertices[SCRATCH_ELEMENTS];
      VertexId input_queue[SCRATCH_ELEMENTS];
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

}  // namespace simplified_advance
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
