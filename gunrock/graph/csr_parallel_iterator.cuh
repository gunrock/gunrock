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

  __host__ __device__ ParallelIterator(const VertexT v, const CsrT* graph)
      : v(v), graph(graph) {
    v_offset = graph->row_offsets[v];
  }

  __host__ __device__ SizeT size() {
    return graph->row_offsets[v + 1] - v_offset;
  }

  __host__ __device__ VertexT neighbor(const SizeT& idx) {
    return graph->column_indices[idx + v_offset];
  }
  __host__ __device__ ValueT value(const SizeT& idx) {
    return graph->edge_values[idx + v_offset];
  }

 private:
  const VertexT v;
  SizeT v_offset;
  const CsrT* graph;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
