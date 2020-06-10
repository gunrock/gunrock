// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * csr_parallel_iterator.cuh
 *
 * @brief CSR Graph Data Structure iterator
 */
#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/csr.cuh>
#include <gunrock/graph/graph_base.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief ParallelIterator iterator for CSR.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG>
struct ParallelIterator<VertexT, SizeT, ValueT, FLAG, HAS_CSR> {
  using CsrT =
      graph::Csr<VertexT, SizeT, ValueT, FLAG & graph::HAS_CSR_MASK,
                 cudaHostRegisterDefault, (FLAG & graph::HAS_CSR) != 0>;
  /**
   * @brief Construct CSR ParallelIterator and store the row offset of the input
   * vertex.
   *
   * @param[in] v Input vertex.
   * @param[in] graph Input graph.
   */
  __host__ __device__ ParallelIterator(const VertexT v, CsrT* graph)
      : v_(v), graph_(graph) {
    v_offset_ = graph_->row_offsets[v_];
  }

  /**
   * @brief Find the number of neighbor of the input vertex.
   *
   * @return Size of vertex neighbors list.
   */
  __host__ __device__ SizeT size() {
    return graph_->row_offsets[v_ + 1] - v_offset_;
  }
  /**
   * @brief Find the neighbor vertex id given an input index.
   *
   * @param[in] idx Input index ranges between 0 and size().
   * @return Neighbor vertex stored at index.
   */
  __host__ __device__ VertexT neighbor(const SizeT& idx) {
    return graph_->column_indices[idx + v_offset_];
  }
  /**
   * @brief Find the neighbor edge value given an input index.
   *
   * @param[in] idx Input index ranges between 0 and size().
   * @return Neighbor edge value stored at index.
   */
  __host__ __device__ ValueT value(const SizeT& idx) {
    return graph_->edge_values[idx + v_offset_];
  }

 private:
  const VertexT v_;
  SizeT v_offset_;
  CsrT* graph_;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
