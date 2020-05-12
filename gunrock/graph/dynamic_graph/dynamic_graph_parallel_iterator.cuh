// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * dynamic_graph_parallel_iterator.cuh
 *
 * @brief Dynamic Graph Data Structure iterator
 */
#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/csr.cuh>
#include <gunrock/graph/graph_base.cuh>

#include <gunrock/graph/dynamic_graph/dynamic_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/dynamic_graph_unweighted.cuh>
#include <gunrock/graph/dynamic_graph/dynamic_graph_weighted.cuh>

namespace gunrock {
namespace graph {

template <typename VertexT, typename ValueT, bool IsMap>
struct nodeReader {
  __device__ static void readNode(const uint32_t& key, VertexT& eId,
                                  ValueT& eVal, const int& laneId) {
    eVal = __shfl_xor_sync(0xFFFFFFFF, key, 1);
    bool key_lane = !(laneId % 2) && (laneId < 30);
    eId = key_lane ? key : EMPTY_KEY;
    eVal = key_lane ? eVal : EMPTY_KEY;
    eId = (key != EMPTY_KEY) ? eId
                             : util::PreDefinedValues<VertexT>::InvalidValue;
    eVal = (key != EMPTY_KEY) ? eVal
                              : util::PreDefinedValues<ValueT>::InvalidValue;
  }
};
template <typename VertexT, typename ValueT>
struct nodeReader<VertexT, ValueT, false> {
  __device__ static void readNode(const uint32_t& key, VertexT& eId,
                                  ValueT& eVal, const int& laneId) {
    bool key_lane = laneId < 30;
    eId = key_lane ? key : EMPTY_KEY;
    bool is_valid = (key != EMPTY_KEY);
    eId = (key != EMPTY_KEY) ? eId
                             : util::PreDefinedValues<VertexT>::InvalidValue;
    eVal = util::PreDefinedValues<ValueT>::InvalidValue;
  }
};

/**
 * @brief ParallelIterator iterator for dynamic graph.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG>
struct ParallelIterator<VertexT, SizeT, ValueT, FLAG, HAS_DYN> {
  using DynT =
      graph::Dyn<VertexT, SizeT, ValueT, FLAG & graph::HAS_DYN_MASK,
                 cudaHostRegisterDefault, (FLAG & graph::HAS_DYN) != 0>;

  __device__ ParallelIterator(VertexT v, DynT* graph)
      : v(v), graph(graph), slab_not_cached(true) {}

  __device__ SizeT size() {
    // count all the number of slabs
    const SizeT v_buckets_offset = graph->dynamicGraph.d_buckets_offset[v];
    const SizeT num_buckets =
        graph->dynamicGraph.d_hash_context[v].getNumBuckets();
    SizeT num_slabs = 0;
    for (int bid = 0; bid < num_buckets; ++bid) {
      SizeT b_edges =
          graph->dynamicGraph.d_edges_per_bucket[bid + v_buckets_offset];
      num_slabs += (b_edges + graph->dynamicGraph.edgesPerSlab - 1) /
                   graph->dynamicGraph.edgesPerSlab;
      num_slabs = b_edges == 0 ? ++num_slabs : num_slabs;
    }
    return num_slabs *= graph->dynamicGraph.keysPerSlab;
  }

  __device__ VertexT neighbor(const SizeT& idx) {
    // tuples are dereferenced starting with base slabs then collision slabs
    SizeT slabId = (idx / graph->dynamicGraph.keysPerSlab);
    SizeT num_buckets = graph->dynamicGraph.d_hash_context[v].getNumBuckets();
    bool is_base_slab = slabId < num_buckets;
    uint32_t laneId = threadIdx.x & 0x1F;
    uint32_t lane_data = EMPTY_KEY;
    if (is_base_slab) {
      lane_data = *(graph->dynamicGraph.d_hash_context[v].getPointerFromBucket(
          slabId, laneId));
    } else {
      // iterate to find the base slab
      slabId -= num_buckets;  // base slabs are dereferenced by now
      SizeT prev_collision_slabs = 0;
      SizeT num_collision_slabs = 0;
      SizeT bid;
      const SizeT v_buckets_offset = graph->dynamicGraph.d_buckets_offset[v];
      for (bid = 0; bid < num_buckets; ++bid) {
        SizeT b_edges =
            graph->dynamicGraph.d_edges_per_bucket[bid + v_buckets_offset];
        num_collision_slabs +=
            (b_edges + graph->dynamicGraph.edgesPerSlab - 1) /
            graph->dynamicGraph.edgesPerSlab;
        num_collision_slabs =
            b_edges == 0 ? ++num_collision_slabs : num_collision_slabs;
        num_collision_slabs--;

        if (slabId >= prev_collision_slabs && slabId < num_collision_slabs) {
          slabId -= prev_collision_slabs;
          break;  // found the base slab
        } else {
          prev_collision_slabs = num_collision_slabs;
        }
      }

      // now iterate to find the slab pointer
      uint32_t ptr = *(
          graph->dynamicGraph.d_hash_context[v].getPointerFromBucket(bid, 31));

      for (SizeT sid = 0; sid < slabId; sid++) {
        ptr = *(
            graph->dynamicGraph.d_hash_context[v].getPointerFromSlab(ptr, 31));
      }
      lane_data = *(graph->dynamicGraph.d_hash_context[v].getPointerFromSlab(
          ptr, laneId));
    }

    nodeReader<VertexT, ValueT, FLAG & HAS_EDGE_VALUES>::readNode(
        lane_data, neighborId, neighborVal, laneId);
    slab_not_cached = false;
    return neighborId;
  }

  __device__ ValueT value(const SizeT& idx) {
    if (slab_not_cached) {
      neighbor(idx);
    }
    return neighborVal;
  }

 private:
  VertexT v;
  DynT* graph;
  bool slab_not_cached;
  VertexT neighborId;
  ValueT neighborVal;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
