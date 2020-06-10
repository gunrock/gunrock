// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * graph_iterator.cuh
 *
 * @brief Graph Data Structure iterator
 */
#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/graph_base.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief ParallelIterator struct to be specialized for different graph data
 * structures
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG,
          GraphFlag GraphType = FLAG& TypeMask>
struct ParallelIterator {
  template <typename GraphT>
  __host__ __device__ ParallelIterator(const VertexT v, GraphT* graph) {}
  __host__ __device__ SizeT size() { return 0; }
  __host__ __device__ VertexT neighbor(const SizeT& idx) { return 0; }
  __host__ __device__ ValueT value(const SizeT& idx) { return 0; }
};
}  // namespace graph
}  // namespace gunrock

#include <gunrock/graph/dynamic_graph/dynamic_graph_parallel_iterator.cuh>
#include <gunrock/graph/csr_parallel_iterator.cuh>

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
