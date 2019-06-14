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
 * @brief kernel plicy for compacted cull filter
 */

#pragma once

#include <cub/cub.cuh>

namespace gunrock {
namespace oprtr {
namespace compacted_cull_filter {

template <typename _Problem, int _CUDA_ARCH, int _MAX_CTA_OCCUPANCY,
          int _LOG_THREADS, int _LOG_GLOBAL_LOAD_SIZE, int _MODE>
struct KernelPolicy {
  typedef _Problem Problem;
  typedef typename Problem::VertexId VertexId;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::Value Value;

  enum {
    MODE = _MODE,
    CUDA_ARCH = _CUDA_ARCH,
    LOG_THREADS = _LOG_THREADS,
    THREADS = 1 << LOG_THREADS,
    MAX_BLOCKS = 1024,
    BLOCK_HASH_BITS = 8,
    BLOCK_HASH_LENGTH = 1 << BLOCK_HASH_BITS,
    BLOCK_HASH_MASK = BLOCK_HASH_LENGTH - 1,
    LOG_GLOBAL_LOAD_SIZE = _LOG_GLOBAL_LOAD_SIZE,
    GLOBAL_LOAD_SIZE = 1 << _LOG_GLOBAL_LOAD_SIZE,
    SHARED_LOAD_SIZE = 8 / sizeof(VertexId),
    WARP_SIZE = GR_WARP_THREADS(CUDA_ARCH),
    LOG_WARP_SIZE = 5,
    WARP_SIZE_MASK = WARP_SIZE - 1,
    WARP_HASH_BITS = 6,
    WARP_HASH_LENGTH = 1 << WARP_HASH_BITS,
    WARP_HASH_MASK = WARP_HASH_LENGTH - 1,
    WARPS = THREADS / WARP_SIZE,
    // ELEMENT_ID_MASK = ~(1ULL << (sizeof(VertexId) * 8 - 2)),
    MAX_CTA_OCCUPANCY = _MAX_CTA_OCCUPANCY,
  };

  // typedef cub::BlockScan<int, KernelPolicy::THREADS, cub::BLOCK_SCAN_RAKING
  // /*cub::BLOCK_SCAN_WARP_SCANS*/> BlockScanT; typedef cub::WarpScan<int>
  // WarpScanT;
  typedef cub::BlockScan<int, THREADS, cub::BLOCK_SCAN_RAKING_MEMOIZE>
      BlockScanT;
  typedef cub::BlockLoad<VertexId*, THREADS, GLOBAL_LOAD_SIZE,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoadT;

  struct SmemStorage {
    VertexId vertices[KernelPolicy::THREADS * KernelPolicy::GLOBAL_LOAD_SIZE];
    // VertexId block_hash[BLOCK_HASH_LENGTH];
    // VertexId warp_hash [WARPS][WARP_HASH_LENGTH];
    // int    temp_space[KernelPolicy::THREADS];
    union {
      // typename cub::BlockLoadT ::TempStorage load_space;
      // typename cub::BlockStoreT::TempStorage store_space;
      typename BlockScanT::TempStorage scan_space;
      // typename WarpScanT::TempStorage scan_space[KernelPolicy::WARPS];
      typename BlockLoadT::TempStorage load_space;
    } cub_storage;

    int num_elements;
    SizeT block_offset;
    SizeT start_segment, end_segment, segment_size;
  };

  enum {
    THREAD_OCCUPANCY = GR_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
    SMEM_OCCUPANCY = GR_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
    CTA_OCCUPANCY = GR_MIN(MAX_CTA_OCCUPANCY,
                           GR_MIN(GR_SM_CTAS(CUDA_ARCH),
                                  GR_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),
    VALID = (CTA_OCCUPANCY > 0),
  };
};

}  // namespace compacted_cull_filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
