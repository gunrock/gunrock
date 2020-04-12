// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * dynamic_graph_base.cuh
 *
 * @brief DYN (Dynamic) Graph Data Structure
 */
#pragma once

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/csr.cuh>

#include <gunrock/graph/dynamic_graph/slabhash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_map.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_set.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief DynamicGraphBase data structure to store basic info about a graph.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename VertexT, typename SizeT, typename ValueT, GraphFlag FLAG>
struct DynamicGraphBase {
 public:
  static constexpr bool REQUIRE_EDGES_VALUES = (FLAG & HAS_EDGE_VALUES) != 0;

  using SlabHashGraphMapT =
      SlabHashGraphMap<VertexT, SizeT, ValueT, REQUIRE_EDGES_VALUES>;
  using SlabHashGraphSetT =
      SlabHashGraphSet<VertexT, SizeT, ValueT, REQUIRE_EDGES_VALUES>;

  // Only one choice now
  using DynamicGraphT =
      typename std::conditional<REQUIRE_EDGES_VALUES, SlabHashGraphMapT,
                                SlabHashGraphSetT>::type;

  DynamicGraphT dynamicGraph;

  cudaError_t Release() {
    dynamicGraph.Release();
    return cudaSuccess;
  }

  bool is_directed;
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
