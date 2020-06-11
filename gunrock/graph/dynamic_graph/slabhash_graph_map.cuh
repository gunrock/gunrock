// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * slabhash_graph_map.cuh
 *
 * @brief Dynamic graph using Concurrent Hash Maps
 */
#pragma once

#include <iostream>

#include <slab_hash.cuh>
#include <cub/cub.cuh>

#include <gunrock/graph/dynamic_graph/slabhash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/kernels/map/insert.cuh>
#include <gunrock/graph/dynamic_graph/kernels/map/delete.cuh>
#include <gunrock/graph/dynamic_graph/kernels/map/helper.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_parallel_iterator.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief HashGraphMap data structure to store an weighted graph using
 * a per-vertex slab hash
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam REQUIRE_VALUES whether the graph is weighted or not
 */
template <typename VertexT, typename SizeT, typename ValueT,
          bool REQUIRE_VALUES,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct SlabHashGraphMap
    : SlabHashGraphBase<VertexT, SizeT, ValueT, REQUIRE_VALUES> {
  using SlabHashGraphParallelIteratorT = SlabHashGraphParallelIterator<
      SlabHashGraphMap<VertexT, SizeT, ValueT, REQUIRE_VALUES,
                       cudaHostRegisterFlag>,
      REQUIRE_VALUES>;

  /**
   * @brief Insert a batch of edges into weighted slab hash graph
   *
   * @param[in] d_edges Device pointer to pairs of edges
   * @param[in] d_values Device pointer to value per pair
   * @param[in] batch_size Size of the inserted batch
   * @param[in] double_batch_edges Double the edges in undirected graph
   */
  template <typename PairT>
  cudaError_t InsertEdgesBatch(PairT* d_edges, ValueT* d_values,
                               SizeT batch_size, bool double_batch_edges) {
    const uint32_t block_size = 128;
    const uint32_t num_blocks = (batch_size + block_size - 1) / block_size;

    slabhash_map_kernels::InsertEdges<<<num_blocks, block_size>>>(
        d_edges, d_values, this->d_hash_context, batch_size,
        this->d_edges_per_node, this->d_edges_per_bucket,
        this->d_buckets_offset, double_batch_edges);
    return cudaSuccess;
  }

  /**
   * @brief Deletes a batch of edges from a weighted slab hash graph
   *
   * @param[in] d_edges Device pointer to pairs of edges
   * @param[in] batch_size Size of the inserted batch
   * @param[in] double_batch_edges Double the edges in undirected graph
   */
  template <typename PairT>
  cudaError_t DeleteEdgesBatch(PairT* d_edges, SizeT batch_size,
                               bool double_batch_edges) {
    const uint32_t block_size = 128;
    const uint32_t num_blocks = (batch_size + block_size - 1) / block_size;

    slabhash_map_kernels::DeleteEdges<<<num_blocks, block_size>>>(
        d_edges, this->d_hash_context, batch_size, this->d_edges_per_node,
        this->d_edges_per_bucket, this->d_buckets_offset, double_batch_edges);
    return cudaSuccess;
  }
  /**
   * @brief Converts CSR to Dynamic graph
   *
   * @param[in] h_row_offsets_csr Host pointer to CSR row offsets
   * @param[in] h_col_indices_csr Host pointer to CSR column indices
   * @param[in] h_edge_values_csr Host pointer to CSR edges values
   * @param[in] num_nodes_csr Number of nodes in the input CSR graph
   * @param[in] is_directed_csr Whether the graph is directed or not
   * @param[in] h_node_values_csr Value per node
   */
  cudaError_t BulkBuildFromCsr(SizeT* h_row_offsets_csr,
                               VertexT* h_col_indices_csr,
                               ValueT* h_edge_values_csr, SizeT num_nodes_csr,
                               bool is_directed_csr,
                               ValueT* h_node_values_csr = nullptr) {
    using PairT = uint2;
    SizeT num_edges_csr = h_row_offsets_csr[num_nodes_csr];
    this->is_directed = is_directed_csr;
    this->InitHashTables(num_nodes_csr, globalLoadFactor, h_row_offsets_csr);

    std::vector<PairT> h_edges_pairs;
    h_edges_pairs.reserve(num_edges_csr);
    for (SizeT v = 0; v < num_nodes_csr; v++) {
      for (SizeT e = h_row_offsets_csr[v]; e < h_row_offsets_csr[v + 1]; e++) {
        h_edges_pairs.push_back(make_uint2(v, h_col_indices_csr[e]));
      }
    }

    PairT* d_edges_pairs;

    CHECK_ERROR(
        cudaMalloc((void**)&d_edges_pairs, sizeof(PairT) * num_edges_csr));
    CHECK_ERROR(cudaMemcpy(d_edges_pairs, h_edges_pairs.data(),
                           sizeof(PairT) * num_edges_csr,
                           cudaMemcpyHostToDevice));

    ValueT* d_edge_values;
    CHECK_ERROR(
        cudaMalloc((void**)&d_edge_values, sizeof(ValueT) * num_edges_csr));
    CHECK_ERROR(cudaMemcpy(d_edge_values, h_edge_values_csr,
                           sizeof(ValueT) * num_edges_csr,
                           cudaMemcpyHostToDevice));

    InsertEdgesBatch(d_edges_pairs, d_edge_values, num_edges_csr, false);

    CHECK_ERROR(cudaFree(d_edge_values));
    CHECK_ERROR(cudaFree(d_edges_pairs));
    return cudaSuccess;
  }

  /**
   * @brief Converts Dynamic graph to CSR
   *
   * @param[out] d_row_offsets Deice pointer to CSR row offsets
   * @param[out] d_col_indices Deice pointer to CSR column indices
   * @param[out] d_edge_values Deice pointer to CSR edges values
   * @param[in] num_nodes_ Number of nodes in the  graph
   * @param[in] num_edges_ Number of edges in the  graph
   * @param[in] d_node_values Device pointer fo value per node
   */
  cudaError_t ToCsr(SizeT* d_row_offsets, VertexT* d_col_indices,
                    ValueT* d_edge_values, SizeT num_nodes_, SizeT num_edges_,
                    ValueT* d_node_values) {
    // Use CUB to determine offsets
    SizeT* d_node_edges_offset;
    CHECK_ERROR(
        cudaMalloc((void**)&d_node_edges_offset, sizeof(SizeT) * num_nodes_));
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  this->d_edges_per_node, d_node_edges_offset,
                                  num_nodes_);
    CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  this->d_edges_per_node, d_node_edges_offset,
                                  num_nodes_);

    const uint32_t block_size = 128;
    const uint32_t num_blocks = (num_nodes_ * 32 + block_size - 1) / block_size;

    slabhash_map_kernels::ToCsr<<<num_blocks, block_size>>>(
        num_nodes_, this->d_hash_context, d_node_edges_offset, d_row_offsets,
        d_col_indices, d_edge_values);

    return cudaSuccess;
  }

  static constexpr float globalLoadFactor = 0.7;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
