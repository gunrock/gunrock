// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * slabhash_graph_set.cuh
 *
 * @brief Dynamic graph using Concurrent Hash Sets
 */
#pragma once

#include <slab_hash.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/kernels/set/insert.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_parallel_iterator.cuh>

#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief HashGraphSet data structure to store an unweighted graph using
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
struct SlabHashGraphSet
    : SlabHashGraphBase<VertexT, SizeT, ValueT, REQUIRE_VALUES> {
  using SlabHashGraphParallelIteratorT = SlabHashGraphParallelIterator<
      SlabHashGraphSet<VertexT, SizeT, ValueT, REQUIRE_VALUES,
                       cudaHostRegisterFlag>,
      REQUIRE_VALUES>;

  /**
   * @brief Insert a batch of edges into unweighted slab hash graph
   *
   * @param[in] d_edges Device pointer to pairs of edges
   * @param[in] batch_size Size of the inserted batch
   * @param[in] double_batch_edges Double the edges in undirected graph
   */
  template <typename PairT>
  cudaError_t InsertEdgesBatch(PairT* d_edges, SizeT batchSize,
                               bool double_batch_edges) {
    return cudaSuccess;
  }

  /**
   * @brief Deletes a batch of edges from an unweighted slab hash graph
   *
   * @param[in] d_edges Device pointer to pairs of edges
   * @param[in] batch_size Size of the inserted batch
   * @param[in] double_batch_edges Double the edges in undirected graph
   */
  template <typename PairT>
  cudaError_t DeleteEdgesBatch(PairT* d_edges, SizeT batch_size,
                               bool double_batch_edges) {
    return cudaSuccess;
  }
  /**
   * @brief Converts CSR to Dynamic graph
   *
   * @param[in] h_row_offsets Host pointer to CSR row offsets
   * @param[in] h_col_indices Host pointer to CSR column indices
   * @param[in] num_nodes_ Number of nodes in the input CSR graph
   * @param[in] is_directed_ Whether the graph is directed or not
   * @param[in] h_node_values Value per node
   */
  cudaError_t BulkBuildFromCsr(SizeT* d_row_offsets, VertexT* d_col_indices,
                               SizeT num_nodes_, bool is_directed_,
                               ValueT* d_node_values = nullptr) {
    this->directed = is_directed_;
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
                    SizeT num_nodes_, SizeT num_edges_,
                    ValueT* d_node_values = nullptr) {
    return cudaSuccess;
  }
};

}  // namespace graph
}  // namespace gunrock
