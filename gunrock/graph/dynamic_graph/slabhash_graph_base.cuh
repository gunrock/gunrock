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
#include <gunrock/graph/dynamic_graph/slabhash_graph_parallel_iterator.cuh>

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
template <typename _VertexT, typename _SizeT, typename _ValueT,
          bool REQUIRE_EDGE_VALUES>
struct SlabHashGraphBase {
  using VertexT = _VertexT;
  using ValueT = _ValueT;
  using SizeT = _SizeT;

  using HashContextT = typename std::conditional<
      REQUIRE_EDGE_VALUES,
      GpuSlabHashContext<VertexT, ValueT, SlabHashTypeT::ConcurrentMap>,
      GpuSlabHashContext<VertexT, ValueT, SlabHashTypeT::ConcurrentSet>>::type;

  static constexpr uint32_t kEdgesPerSlab = REQUIRE_EDGE_VALUES ? 15 : 30;

  DynamicAllocatorT* memory_allocator;

  HashContextT* h_hash_context;
  HashContextT* d_hash_context;

  SizeT* buckets_per_table;

  SizeT* d_edges_per_node;
  int8_t* d_base_slabs;

  SizeT* d_edges_per_bucket;
  SizeT* d_buckets_offset;

  SlabHashGraphBase(){};
  // todo: add node values
  /**
   * @brief Allocate maximum capacity memory for SlabHash graph .
   *
   * @param[in] max_nodes Maximum number of nodes that the graph can store
   */
  void Allocate(SizeT max_nodes = 1 << 20) {
    memory_allocator = new DynamicAllocatorT;

    nodes_capacity = max_nodes;
    buckets_per_table = new SizeT[nodes_capacity];

    std::memset(buckets_per_table, 0, sizeof(SizeT) * nodes_capacity);

    h_hash_context = new HashContextT[nodes_capacity];

    CHECK_ERROR(cudaMalloc((void**)&d_hash_context,
                           sizeof(HashContextT) * nodes_capacity));
    CHECK_ERROR(
        cudaMalloc((void**)&d_edges_per_node, sizeof(SizeT) * nodes_capacity));
    CHECK_ERROR(
      cudaMemset(d_edges_per_node, 0, sizeof(SizeT) * nodes_capacity));
  }

  ~SlabHashGraphBase() {}

  /**
   * @brief Initialize the dynamic graph for a number of nodes.
   *
   * @param[in] num_init_nodes Number of nodes to initialize the dynamic graph
   * for.
   * @param[in] load_factor Load factor per hash table
   * @param[in] h_row_offsets_hint Optional hint array to a row offset CSR style
   * array for hints on number of edges per node. If not provided each hash
   * table has a single bucket.
   */
  cudaError_t InitHashTables(SizeT num_init_nodes, float load_factor,
                             SizeT* h_row_offsets_hint = nullptr) {
    assert(num_init_nodes <= nodes_capacity &&
           "Capcity lower than required number of nodes");
    SizeT total_num_buckets = 0;
    num_nodes = num_init_nodes;

    // Compute the required number of base slabs
    for (SizeT i = 0; i < num_nodes; i++) {
      SizeT node_edges;
      if (h_row_offsets_hint) {
        node_edges = h_row_offsets_hint[i + 1] - h_row_offsets_hint[i];
      } else {
        node_edges = 0;
      }
      buckets_per_table[i] =
          ceil(node_edges / (load_factor * float(kEdgesPerSlab)));
      buckets_per_table[i] = std::max(buckets_per_table[i], SizeT(1));
      total_num_buckets += buckets_per_table[i];
    }
    buckets_capacity = total_num_buckets;

    // Allocate memory for base slabs
    CHECK_ERROR(cudaMalloc((void**)&d_edges_per_bucket,
    sizeof(SizeT) * buckets_capacity));
    CHECK_ERROR(cudaMalloc((void**)&d_buckets_offset,
        sizeof(SizeT) * buckets_capacity));

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

    std::vector<SizeT> h_buckets_offset(buckets_capacity);

    uint32_t hash_func_x, hash_func_y;
    std::mt19937 rng(time(0));
    hash_func_x = rng() % kPrimeDivisor;
    if (hash_func_x < 1) hash_func_x = 1;
    hash_func_y = rng() % kPrimeDivisor;

    using SlabT = typename ConcurrentMapT<VertexT, ValueT>::SlabTypeT;
    SlabT* d_base_slabs32 = reinterpret_cast<SlabT*>(d_base_slabs);

    // initialize slab hash contexts
    SizeT current_base_buckets_offset = 0;
    for (SizeT i = 0; i < num_nodes; i++) {
      h_hash_context[i].initParameters(
          buckets_per_table[i], hash_func_x, hash_func_y,
          reinterpret_cast<int8_t*>(&d_base_slabs32[current_base_buckets_offset]),
          memory_allocator->getContextPtr());
      h_buckets_offset[i] = current_base_buckets_offset;
      current_base_buckets_offset += buckets_per_table[i];
    }

    CHECK_ERROR(cudaMemcpy(d_hash_context, h_hash_context,
                           sizeof(HashContextT) * num_nodes,
                           cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMemcpy(d_buckets_offset, h_buckets_offset.data(),
                           sizeof(SizeT) * buckets_capacity,
                           cudaMemcpyHostToDevice));
    return cudaSuccess;
  }

  /**
   * @brief Query a graph vertex neghbor's count
   *
   * @param[in] v Query vertex
   * @return v's neghbor's count
   */
  __device__ __forceinline__ SizeT GetNeighborsCount(const VertexT& v) const {
    return d_edges_per_node[v];
  }

  /**
   * @brief Query the graph maximun nodes capacity
   *
   * @param[out] the graph maximum nodes capacity
   */
  SizeT GetNodesCapacity() const { return nodes_capacity; }

  /**
   * @brief Query the graph maximun buckets capacity
   *
   * @param[out] the graph maximum buckets capacity
   */
  SizeT GetSlabsCapacity() const { return buckets_capacity; }

  /**
   * @brief Extend the capcity of the graph vertices
   *
   */
  cudaError_t ExtendCapacity() { return cudaSuccess; }

  cudaError_t Release() {
    delete[] h_hash_context;
    delete[] buckets_per_table;
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

  static constexpr uint32_t kPrimeDivisor = 4294967291u;
  static constexpr uint32_t kKeysPerSlab = 32;
};
}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
