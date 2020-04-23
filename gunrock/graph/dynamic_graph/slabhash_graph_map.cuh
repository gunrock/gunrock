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
   * @param[in] h_row_offsets Host pointer to CSR row offsets
   * @param[in] h_col_indices Host pointer to CSR column indices
   * @param[in] h_edge_values Host pointer to CSR edges values
   * @param[in] num_nodes_ Number of nodes in the input CSR graph
   * @param[in] is_directed_ Whether the graph is directed or not
   * @param[in] h_node_values Value per node
   */
  cudaError_t BulkBuildFromCsr(SizeT* h_row_offsets, VertexT* h_col_indices,
                               ValueT* h_edge_values, SizeT num_nodes_,
                               bool is_directed_,
                               ValueT* h_node_values = nullptr) {
    using PairT = uint2;
    SizeT num_edges_ = h_row_offsets[num_nodes_];
    PairT* d_edges_pairs;
    this->is_directed = is_directed_;
    this->Init(h_row_offsets, num_nodes_, num_edges_, edgesPerSlab,
               globalLoadFactor, h_col_indices, d_edges_pairs);

    ValueT* d_edge_values;
    CHECK_ERROR(
        cudaMalloc((void**)&d_edge_values, sizeof(ValueT) * num_edges_));
    CHECK_ERROR(cudaMemcpy(d_edge_values, h_edge_values,
                           sizeof(ValueT) * num_edges_,
                           cudaMemcpyHostToDevice));

    InsertEdgesBatch(d_edges_pairs, d_edge_values, num_edges_, false);

    CHECK_ERROR(cudaFree(d_edge_values));
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

  static constexpr uint32_t edgesPerSlab = 15;
  static constexpr float globalLoadFactor = 0.7;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
