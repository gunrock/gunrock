// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * delete.cuh
 *
 * @brief SlabHash graph Graph Data Structure map deleteion kernels
 */
#pragma once

#include <slab_hash.cuh>

namespace gunrock {
namespace graph {
namespace slabhash_map_kernels {
template <typename PairT, typename SizeT, typename ContextT>
__device__ void deleteWarpEdges(PairT& thread_edge, ContextT*& hashContexts,
                                SizeT*& d_edges_per_vertex,
                                SizeT*& d_edges_per_bucket,
                                SizeT*& d_buckets_offset, uint32_t& laneId,
                                bool to_delete) {
  uint32_t dst_bucket;

  if (to_delete) {
    dst_bucket = hashContexts[thread_edge.x].computeBucket(thread_edge.y);
  }

  uint32_t work_queue;
  while (work_queue = __ballot_sync(0xFFFFFFFF, to_delete)) {
    uint32_t cur_lane = __ffs(work_queue) - 1;
    uint32_t cur_src = __shfl_sync(0xFFFFFFFF, thread_edge.x, cur_lane, 32);
    bool same_src = (cur_src == thread_edge.x) && to_delete;

    if (same_src) {
      SizeT bucket_offset = d_buckets_offset[cur_src];
      atomicSub(&d_edges_per_bucket[bucket_offset + dst_bucket], 1);
    }
    to_delete &= !same_src;
    bool success = hashContexts[cur_src].deleteKey(same_src, laneId,
                                                   thread_edge.y, dst_bucket);

    uint32_t deleted_count = __popc(__ballot_sync(0xFFFFFFFF, success));
    if (laneId == 0) {
      atomicSub(&d_edges_per_vertex[cur_src], deleted_count);
    }
  }
}
template <typename PairT, typename SizeT, typename ContextT>
__global__ void DeleteEdges(PairT* d_edges, ContextT* hashContexts,
                            SizeT num_edges, SizeT* d_edges_per_vertex,
                            SizeT* d_edges_per_bucket, SizeT* d_buckets_offset,
                            bool make_batch_undirected) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_edges) {
    return;
  }

  PairT thread_edge;
  bool to_delete = false;
  if (tid < num_edges) {
    thread_edge = d_edges[tid];
    to_delete = (thread_edge.x != thread_edge.y);
  }

  deleteWarpEdges(thread_edge, hashContexts, d_edges_per_vertex,
                  d_edges_per_bucket, d_buckets_offset, laneId, to_delete);

  if (make_batch_undirected) {
    PairT reverse_edge = make_uint2(thread_edge.y, thread_edge.x);
    deleteWarpEdges(reverse_edge, hashContexts, d_edges_per_vertex,
                    d_edges_per_bucket, d_buckets_offset, laneId, to_delete);
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
