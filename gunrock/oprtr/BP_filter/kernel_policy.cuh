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

#include <gunrock/util/scan/block_scan.cuh>

namespace gunrock {
namespace oprtr {
namespace BP {

template <
    // typename _VertexT,      // Data types
    // typename _InKeyT,
    typename _SizeT,
    // typename _ValueT,
    int _MAX_CTA_OCCUPANCY,  // Tunable parameters
    int _LOG_THREADS, int _LOG_BLOCKS>
struct KernelPolicy {
  //---------------------------------------------------------------------
  // Constants and typedefs
  //---------------------------------------------------------------------

  // typedef _VertexT  VertexT;
  // typedef _InKeyT   InKeyT;
  typedef _SizeT SizeT;
  // typedef _ValueT   ValueT;

  enum {
    // CUDA_ARCH                       = _CUDA_ARCH,
    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    LOG_BLOCKS = _LOG_BLOCKS,
    BLOCKS = 1 << LOG_BLOCKS,
  };
  typedef util::Block_Scan<SizeT, LOG_THREADS> BlockScanT;

  /**
   * @brief Shared memory storage type for the CTA
   */
  struct SmemStorage {
    // Scratch elements
    struct {
      typename BlockScanT::Temp_Space scan_space;
      SizeT block_offset;
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

}  // namespace BP
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
