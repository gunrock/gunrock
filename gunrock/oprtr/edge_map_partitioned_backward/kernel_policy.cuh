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

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned_backward {

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
 * @tparam _LOG_BLOCKS                  Number of blocks per grid (log).
 * @tparam _LIGHT_EDGE_THRESHOLD        When to switch between two edge relax
 * algorithms
 */
template <typename _ProblemData,
          // Machine parameters
          int _CUDA_ARCH,
          // Behavioral control parameters
          // bool _INSTRUMENT,
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

  enum {

    CUDA_ARCH = _CUDA_ARCH,
    // INSTRUMENT                      = _INSTRUMENT,

    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
    LIGHT_EDGE_THRESHOLD = _LIGHT_EDGE_THRESHOLD,
  };

  /**
   * @brief Shared memory storage type for the CTA
   */
  struct SmemStorage {
    enum {
      // Amount of storage we can use for hashing scratch space under target
      // occupancy
      MAX_SCRATCH_BYTES_PER_CTA =
          (GR_SMEM_BYTES(CUDA_ARCH) / _MIN_CTA_OCCUPANCY) -
          128,  // Fudge-factor to guarantee occupancy

      SCRATCH_ELEMENT_SIZE = sizeof(SizeT) + sizeof(VertexId) * 2,

      SCRATCH_ELEMENTS =
          (THREADS > MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE)
              ? MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE
              : THREADS,
    };

    // Scratch elements
    struct {
      SizeT s_edges[SCRATCH_ELEMENTS];
      VertexId s_vertices[SCRATCH_ELEMENTS];
      VertexId s_edge_ids[SCRATCH_ELEMENTS];
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

}  // namespace edge_map_partitioned_backward
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
