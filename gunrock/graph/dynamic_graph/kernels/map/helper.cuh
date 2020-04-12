// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * helper.cuh
 *
 * @brief SlabHash graph Graph Data Structure map kernels
 */
#pragma once

#include <slab_hash.cuh>

namespace gunrock {
namespace graph {
namespace slabhash_map_kernels {
template <typename VertexT, typename ValueT, typename SizeT, typename ContextT>
__global__ void ToCsr(SizeT num_nodes, ContextT* hashContexts,
                      SizeT* d_node_edges_offset, SizeT* d_row_offsets,
                      VertexT* d_col_indices, ValueT* d_edge_values) {
  using SlabHashT = ConcurrentMapT<VertexT, VertexT>;

  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;
  uint32_t vid = tid >> 5;

  if (vid >= num_nodes) return;

  uint32_t num_buckets = hashContexts[vid].getNumBuckets();
  uint32_t cur_offset = 0;

  if (vid != 0) cur_offset = d_node_edges_offset[vid - 1];

  if (laneId == 0) d_row_offsets[vid] = cur_offset;

  if (vid == (num_nodes - 1)) d_row_offsets[vid + 1] = d_node_edges_offset[vid];

  uint lanemask;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask));

  for (int i = 0; i < num_buckets; i++) {
    uint32_t next = SlabHashT::A_INDEX_POINTER;
    do {
      uint32_t key =
          (next == SlabHashT::A_INDEX_POINTER)
              ? *(hashContexts[vid].getPointerFromBucket(i, laneId))
              : *(hashContexts[vid].getPointerFromSlab(next, laneId));

      next = __shfl_sync(0xFFFFFFFF, key, 31, 32);

      uint32_t val = __shfl_xor_sync(0xFFFFFFFF, key, 1);

      bool key_lane = !(laneId % 2) && (laneId < 30);

      key = key_lane ? key : EMPTY_KEY;
      val = key_lane ? val : EMPTY_KEY;

      bool is_valid = (key != EMPTY_KEY);
      uint32_t rank_bmp = __ballot_sync(0xFFFFFFFF, is_valid);
      uint32_t rank = __popc(rank_bmp & lanemask);
      uint32_t sum = __popc(rank_bmp);

      uint32_t lane_offset = cur_offset + rank;
      if (is_valid) {
        d_col_indices[lane_offset] = key;
        d_edge_values[lane_offset] = val;
      }
      cur_offset += sum;

    } while (next != SlabHashT::EMPTY_INDEX_POINTER);
  }
}
}  // namespace slabhash_map_kernels
}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
