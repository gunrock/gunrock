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

/**
 * @brief ParallelIterator iterator for dynamic graph.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG>
struct ParallelIterator<VertexT, SizeT, ValueT, FLAG, HAS_DYN>
    : graph::Dyn<VertexT, SizeT, ValueT, FLAG & graph::HAS_DYN_MASK,
                 cudaHostRegisterDefault,
                 (FLAG & graph::HAS_DYN) != 0>::DynamicGraphParallelIterator {
  using DynT =
      graph::Dyn<VertexT, SizeT, ValueT, FLAG & graph::HAS_DYN_MASK,
                 cudaHostRegisterDefault, (FLAG & graph::HAS_DYN) != 0>;

  __device__ ParallelIterator(const VertexT v, DynT* graph)
      : DynT::DynamicGraphParallelIterator(v, &graph->dynamicGraph) {}
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
