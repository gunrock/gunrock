// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * slabhash_graph_parallel_iterator.cuh
 *
 * @brief Dynamic Graph Data Structure iterator.
 */
#pragma once

namespace gunrock {
namespace graph {

/**
 * @brief slabReader structure reads and decodes a slab hash map's slab.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam ValueT  Value identifier type.
 * @tparam IsMap Boolean for map or set specialization (true).
 */
template <typename VertexT, typename ValueT, bool IsMap>
struct slabReader {
  /**
   * @brief Reads a slab hash key then return a list of the valid edges and
   * values and InvalidValue for the invalid ones.
   *
   * @param[in] key Input key/value stored in a slab.
   * @param[out] e_dst Output edge destinaion.
   * @param[in] e_val Output edge value.
   * @param[in] lane_id Lane id in the warp.
   */
  __device__ static void readSlab(const uint32_t& key, VertexT& e_dst,
                                  ValueT& e_val, const int& lane_id) {
    // shuffle values to be in even lanes
    e_val = __shfl_xor_sync(0xFFFFFFFF, key, 1);
    // even lanes are valid and last two lanes are reserved
    bool key_lane = !(lane_id % 2) && (lane_id < 30);
    e_dst = key_lane ? key : EMPTY_KEY;
    e_val = key_lane ? e_val : EMPTY_KEY;
    e_dst = (key != EMPTY_KEY) ? e_dst
                               : util::PreDefinedValues<VertexT>::InvalidValue;
    e_val = (key != EMPTY_KEY) ? e_val
                               : util::PreDefinedValues<ValueT>::InvalidValue;
  }
};
/**
 * @brief slabReader specialized structure reads and decodes a slab hash set's
 * slab.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam ValueT  Value identifier type.
 * @tparam IsMap False.
 */
template <typename VertexT, typename ValueT>
struct slabReader<VertexT, ValueT, false> {
  /**
   * @brief Reads a slab hash key then return a list of the valid edges and
   * InvalidValue for the invalid ones.
   *
   * @param[in] key Input key stored in a slab.
   * @param[out] e_dst Output edge destinaion.
   * @param[in] e_val Output edge value (InvalidValue for a set).
   * @param[in] lane_id Lane id in the warp.
   */
  __device__ static void readSlab(const uint32_t& key, VertexT& e_id,
                                  ValueT& e_val, const int& lane_id) {
    // only the two last lanes are invalid (reserved).
    bool key_lane = lane_id < 30;
    e_id = key_lane ? key : EMPTY_KEY;
    bool is_valid = (key != EMPTY_KEY);
    e_id = (key != EMPTY_KEY) ? e_id
                              : util::PreDefinedValues<VertexT>::InvalidValue;
    e_val = util::PreDefinedValues<ValueT>::InvalidValue;
  }
};

/**
 * @brief ParallelIterator iterator for slab hash dynamic graph.
 *
 * @tparam GraphT Graph identifer type.
 * @tparam isMap Boolean for map or set specialization.
 */
template <typename GraphT, bool isMap>
struct SlabHashGraphParallelIterator {
  using VertexT = typename GraphT::VertexT;
  using ValueT = typename GraphT::ValueT;
  using SizeT = typename GraphT::SizeT;

  /**
   * @brief Iterator constructor given vertex in a graph.
   *
   * @param[in] v Input Vertex to iterate around.
   * @param[in] graph Input graph.
   */
  __device__ SlabHashGraphParallelIterator(const VertexT v, GraphT* graph)
      : v_(v), graph_(graph), slab_not_cached_(true) {}

  /**
   * @brief Counts the size (capacity) of all edges associated with the input
   * vertex.
   *
   * @return Size (capacity) of the vertex.
   */
  __device__ SizeT size() {
    // count all the number of slabs
    const SizeT v_buckets_offset = graph_->d_buckets_offset[v_];
    const SizeT num_buckets = graph_->d_hash_context[v_].getNumBuckets();
    SizeT num_slabs = 0;
    for (int bid = 0; bid < num_buckets; ++bid) {
      SizeT b_edges = graph_->d_edges_per_bucket[bid + v_buckets_offset];
      // compute the number of slabs in this bucket
      num_slabs +=
          (b_edges + graph_->kEdgesPerSlab - 1) / graph_->kEdgesPerSlab;
      // Make sure we have a minimum of 1 slab. Base slab is allocated if there
      // are no edges stored
      num_slabs = b_edges == 0 ? ++num_slabs : num_slabs;
    }
    // Capacity is the total number of slabs * the total number of keys per
    // slab
    return num_slabs *= graph_->kKeysPerSlab;
  }

  /**
   * @brief Find the neighbor vertex id given an input index.
   *
   * @param[in] idx Input index ranges between 0 and size().
   * @return Neighbor vertex stored at index.
   */
  __device__ VertexT neighbor(const SizeT& idx) {
    // tuples are dereferenced starting with base slabs then collision slabs.
    // Collision slabs are indexed starting from the first base slab
    // Example:
    // |Base|->[Collision]
    // | 0 |->[ 3 ] -> [ 4 ]
    // | 1 |->[ 5 ]
    // | 2 |->[ 6 ] -> [ 7 ]

    SizeT slab_id = (idx / graph_->kKeysPerSlab);
    SizeT num_buckets = graph_->d_hash_context[v_].getNumBuckets();
    bool is_base_slab = slab_id < num_buckets;
    uint32_t lane_id = threadIdx.x & 0x1F;
    uint32_t lane_data = EMPTY_KEY;
    if (is_base_slab) {
      lane_data =
          *(graph_->d_hash_context[v_].getPointerFromBucket(slab_id, lane_id));
    } else {
      // linear search iteration to find the base slab
      slab_id -= num_buckets;  // base slabs are dereferenced first
      SizeT prev_collision_slabs = 0;
      SizeT num_collision_slabs = 0;
      SizeT bid;
      const SizeT v_buckets_offset = graph_->d_buckets_offset[v_];
      for (bid = 0; bid < num_buckets; ++bid) {
        SizeT b_edges = graph_->d_edges_per_bucket[bid + v_buckets_offset];
        // increment by counting all the number of slabs for this bucket
        num_collision_slabs +=
            (b_edges + graph_->kEdgesPerSlab - 1) / graph_->kEdgesPerSlab;
        // decrement the number of slabs by one to account for base slab. Only
        // do so if the number of edges is greater than one (i.e. previous
        // increment accounted for the base slab)
        num_collision_slabs -= (b_edges != 0);

        if (slab_id >= prev_collision_slabs && slab_id < num_collision_slabs) {
          // slab id in collision slabs
          slab_id -= prev_collision_slabs;
          break;  // found the base slab
        } else {
          prev_collision_slabs = num_collision_slabs;
        }
      }

      // now iterate to find the slab pointer
      uint32_t ptr =
          *(graph_->d_hash_context[v_].getPointerFromBucket(bid, 31));

      for (SizeT sid = 0; sid < slab_id; sid++) {
        ptr = *(graph_->d_hash_context[v_].getPointerFromSlab(ptr, 31));
      }
      lane_data =
          *(graph_->d_hash_context[v_].getPointerFromSlab(ptr, lane_id));
    }

    slabReader<VertexT, ValueT, isMap>::readSlab(lane_data, neighbor_id_,
                                                 neighbor_val_, lane_id);
    slab_not_cached_ = false;
    return neighbor_id_;
  }
  /**
   * @brief Find the neighbor edge value given an input index.
   *
   * @param[in] idx Input index ranges between 0 and size().
   * @return Neighbor edge value stored at index.
   */
  __device__ ValueT value(const SizeT& idx) {
    if (slab_not_cached_) {
      neighbor(idx);
    }
    return neighbor_val_;
  }

 private:
  const VertexT v_;
  GraphT* graph_;
  bool slab_not_cached_;
  VertexT neighbor_id_;
  ValueT neighbor_val_;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
