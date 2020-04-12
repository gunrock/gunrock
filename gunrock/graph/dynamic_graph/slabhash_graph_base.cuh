// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * slabhash_graph_base.cuh
 *
 * @brief Hash graph Graph Data Structure
 */
#pragma once

#include <vector>

#include <slab_hash.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief SlabHashGraphBase data structure to store basic info about a graph
 * and allocate the required GPU memory
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam REQUIRE_EDGE_VALUES whether the graph is weighted or not
 */
template <typename VertexT, typename SizeT, typename ValueT,
          bool REQUIRE_EDGE_VALUES>
struct SlabHashGraphBase {
  using HashContextT = typename std::conditional<
      REQUIRE_EDGE_VALUES,
      GpuSlabHashContext<VertexT, ValueT, SlabHashTypeT::ConcurrentMap>,
      GpuSlabHashContext<VertexT, ValueT, SlabHashTypeT::ConcurrentSet>>::type;

  DynamicAllocatorT* memory_allocator;

  HashContextT* h_hash_context;
  HashContextT* d_hash_context;

  std::vector<SizeT> buckets_per_table;

  SizeT* d_edges_per_node;
  int8_t* d_base_slabs;

  SizeT* d_edges_per_bucket;
  SizeT* d_buckets_offset;

  // todo: add node values
  /**
   * @brief Allocate maximum capacity memory for SlabHash graph.
   *
   * @param[in] max_nodes Maximum number of nodes that the graph can store
   * @param[in] max_buckets Maximum number of buckets that the graph will use
   */
  SlabHashGraphBase(SizeT max_nodes = 1 << 20, SizeT max_buckets = 1 << 25) {
    memory_allocator = new DynamicAllocatorT;

    nodes_capacity = max_nodes;
    buckets_capacity = max_buckets;
    buckets_per_table.resize(nodes_capacity);
    std::fill(buckets_per_table.begin(), buckets_per_table.end(), 0);

    h_hash_context = new HashContextT[nodes_capacity];

    CHECK_ERROR(cudaMalloc((void**)&d_hash_context,
                           sizeof(HashContextT) * nodes_capacity));
    CHECK_ERROR(
        cudaMalloc((void**)&d_edges_per_node, sizeof(SizeT) * nodes_capacity));
    CHECK_ERROR(cudaMalloc((void**)&d_edges_per_bucket,
                           sizeof(SizeT) * buckets_capacity));
    CHECK_ERROR(cudaMalloc((void**)&d_buckets_offset,
                           sizeof(SizeT) * buckets_capacity));

    CHECK_ERROR(
        cudaMemset(d_edges_per_node, 0, sizeof(SizeT) * nodes_capacity));
    CHECK_ERROR(
        cudaMemset(d_edges_per_bucket, 0, sizeof(SizeT) * buckets_capacity));
    CHECK_ERROR(
        cudaMemset(d_buckets_offset, 0, sizeof(SizeT) * buckets_capacity));

    size_t slab_unit_size =
        GpuSlabHashContext<VertexT, ValueT,
                           SlabHashTypeT::ConcurrentMap>::getSlabUnitSize();
    CHECK_ERROR(
        cudaMalloc((void**)&d_base_slabs, slab_unit_size * buckets_capacity));
    CHECK_ERROR(
        cudaMemset(d_base_slabs, 0xFF, slab_unit_size * buckets_capacity));
  }

  ~SlabHashGraphBase() {}

  /**
   * @brief Converts CSR to Dynamic graph and Allocate GPU memory for input
   * pairs
   *
   * @param[in] h_row_offsets host pointer to CSR row offsets
   * @param[in] num_nodes_ number of nodes in the input CSR graph
   * @param[in] num_edges_ number of edges in the input CSR graph
   * @param[in] edges_per_slab number of edges per slab (15 for hashmap, 31 for
   * hashset)
   * @param[in] loadfactor load factor per hash table
   * @param[in] h_col_indices host pointer to CSR column indices
   * @param[in] d_edges_pairs device pointer to COO generated edge pairs
   */
  template <typename PairT>
  cudaError_t Init(SizeT* h_row_offsets, SizeT num_nodes_, SizeT num_edges_,
                   SizeT edges_per_slab, float load_factor,
                   VertexT* h_col_indices, PairT*& d_edges_pairs) {
    assert(num_nodes_ < nodes_capacity &&
           "Capcity lower than required number of nodes");
    SizeT total_num_buckets = 0;
    SizeT total_num_edges = 0;
    num_nodes = num_nodes_;

    std::vector<PairT> h_edges_pairs(num_edges_);
    std::vector<SizeT> h_buckets_offset(buckets_capacity);

    uint32_t hash_func_x, hash_func_y;
    std::mt19937 rng(time(0));
    hash_func_x = rng() % PRIME_DIVISOR_;
    if (hash_func_x < 1) hash_func_x = 1;
    hash_func_y = rng() % PRIME_DIVISOR_;

    using SlabT = typename ConcurrentMapT<VertexT, ValueT>::SlabTypeT;
    SlabT* d_base_slabs32 = reinterpret_cast<SlabT*>(d_base_slabs);

    for (SizeT i = 0; i < num_nodes; i++) {
      SizeT node_edges = h_row_offsets[i + 1] - h_row_offsets[i];
      buckets_per_table[i] =
          ceil(node_edges / (load_factor * float(edges_per_slab)));
      buckets_per_table[i] = std::max(buckets_per_table[i], SizeT(1));
      for (VertexT v = h_row_offsets[i]; v < h_row_offsets[i + 1]; v++) {
        h_edges_pairs[total_num_edges] = make_uint2(i, h_col_indices[v]);
        total_num_edges++;
      }

      assert(total_num_buckets < buckets_capacity &&
             "Capcity lower than required number of base slabs");

      h_hash_context[i].initParameters(
          buckets_per_table[i], hash_func_x, hash_func_y,
          reinterpret_cast<int8_t*>(&d_base_slabs32[total_num_buckets]),
          memory_allocator->getContextPtr());
      h_buckets_offset[i] = total_num_buckets;
      total_num_buckets += buckets_per_table[i];
    }
    assert(total_num_edges == num_edges_ &&
           "Edges in the graph mismatch the expected count");

    CHECK_ERROR(cudaMalloc((void**)&d_edges_pairs, sizeof(PairT) * num_edges_));
    CHECK_ERROR(cudaMemcpy(d_edges_pairs, h_edges_pairs.data(),
                           sizeof(PairT) * num_edges_, cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMemcpy(d_hash_context, h_hash_context,
                           sizeof(HashContextT) * num_nodes,
                           cudaMemcpyHostToDevice));
    return cudaSuccess;
  }

  /**
   * @brief Query the graph maximun nodes capacity
   *
   * @param[out] the graph maximum nodes capacity
   */
  SizeT GetNodesCapacity() { return nodes_capacity; }

  /**
   * @brief Query the graph maximun buckets capacity
   *
   * @param[out] the graph maximum buckets capacity
   */
  SizeT GetSlabsCapacity() { return buckets_capacity; }

  /**
   * @brief Extend the capcity of the graph vertices
   *
   */
  cudaError_t ExtendCapacity() { return cudaSuccess; }

  cudaError_t Release() {
    delete[] h_hash_context;
    delete memory_allocator;

    cudaFree(d_hash_context);
    cudaFree(d_edges_per_node);
    cudaFree(d_base_slabs);
    cudaFree(d_edges_per_bucket);
    cudaFree(d_buckets_offset);
    return cudaSuccess;
  }

  bool is_directed;

  SizeT num_nodes;
  SizeT num_edges;

  SizeT nodes_capacity;
  SizeT buckets_capacity;

  static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;
};
}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
