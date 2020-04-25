// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * insert.cuh
 *
 * @brief SlabHash graph Graph Data Structure set insertion kernel
 */
#pragma once

#include <slab_hash.cuh>

namespace gunrock {
namespace graph {
namespace slabhash_set_kernels {
template <typename PairT, typename SizeT, typename ContextT>
__device__ void insertWarpEdges(PairT& thread_edge, ContextT*& hashContexts,
                                SizeT*& d_edges_per_vertex,
                                SizeT*& d_edges_per_bucket,
                                SizeT*& d_buckets_offset, uint32_t& laneId,
                                bool to_insert,
                                AllocatorContextT& local_allocator_ctx) {
  uint32_t dst_bucket;

  if (to_insert) {
    dst_bucket = hashContexts[thread_edge.x].computeBucket(thread_edge.y);
  }

  uint32_t work_queue;
  while (work_queue = __ballot_sync(0xFFFFFFFF, to_insert)) {
    uint32_t cur_lane = __ffs(work_queue) - 1;
    uint32_t cur_src = __shfl_sync(0xFFFFFFFF, thread_edge.x, cur_lane, 32);
    bool same_src = (cur_src == thread_edge.x) && to_insert;

    to_insert &= !same_src;
    bool success = hashContexts[cur_src].insertPairUnique(
        same_src, laneId, thread_edge.y, dst_bucket, local_allocator_ctx);
    uint32_t added_count = __popc(__ballot_sync(0xFFFFFFFF, success));

    if (laneId == 0) {
      atomicAdd(&d_edges_per_vertex[cur_src], added_count);
    }
    if (success) {
      SizeT bucket_offset = d_buckets_offset[cur_src];
      atomicAdd(&d_edges_per_bucket[bucket_offset + dst_bucket], 1);
    }
  }
}
template <typename PairT, typename SizeT, typename ContextT>
__global__ void InsertEdges(PairT* d_edges, ContextT* hashContexts,
                            SizeT num_edges, SizeT* d_edges_per_vertex,
                            SizeT* d_edges_per_bucket, SizeT* d_buckets_offset,
                            bool make_batch_undirected) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_edges) {
    return;
  }

  PairT thread_edge;
  bool to_insert = false;
  if (tid < num_edges) {
    thread_edge = d_edges[tid];
    to_insert = (thread_edge.x != thread_edge.y);
  }

  AllocatorContextT local_allocator_ctx(hashContexts[0].getAllocatorContext());
  local_allocator_ctx.initAllocator(tid, laneId);

  insertWarpEdges(thread_edge, hashContexts, d_edges_per_vertex,
                  d_edges_per_bucket, d_buckets_offset, laneId, to_insert,
                  local_allocator_ctx);

  if (make_batch_undirected) {
    PairT reverse_edge = make_uint2(thread_edge.y, thread_edge.x);
    insertWarpEdges(reverse_edge, hashContexts, d_edges_per_vertex,
                    d_edges_per_bucket, d_buckets_offset, laneId, to_insert,
                    local_allocator_ctx);
  }
}
}  // namespace slabhash_set_kernels
}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
