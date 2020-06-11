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
#include <gunrock/graph/dynamic_graph/slabhash_graph_parallel_iterator.cuh>

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
template <typename _VertexT, typename _SizeT, typename _ValueT, GraphFlag _FLAG>
struct DynamicGraphBase {
 public:
  using VertexT = _VertexT;
  using ValueT = _ValueT;
  using SizeT = _SizeT;

  static constexpr GraphFlag FLAG = _FLAG;
  static constexpr bool REQUIRE_EDGES_VALUES = (FLAG & HAS_EDGE_VALUES) != 0;

  using SlabHashGraphMapT =
      SlabHashGraphMap<VertexT, SizeT, ValueT, REQUIRE_EDGES_VALUES>;
  using SlabHashGraphSetT =
      SlabHashGraphSet<VertexT, SizeT, ValueT, REQUIRE_EDGES_VALUES>;

  // Only one choice now
  using DynamicGraphT =
      typename std::conditional<REQUIRE_EDGES_VALUES, SlabHashGraphMapT,
                                SlabHashGraphSetT>::type;
  using DynamicGraphParallelIterator =
      typename DynamicGraphT::SlabHashGraphParallelIteratorT;

  DynamicGraphT dynamicGraph;

  cudaError_t Allocate(SizeT max_nodes) {
    dynamicGraph.Allocate(max_nodes);
    return cudaSuccess;
  }
  cudaError_t Release() {
    dynamicGraph.Release();
    return cudaSuccess;
  }
  __device__ __forceinline__ SizeT
  GetNeighborListLength(const VertexT &v) const {
    return dynamicGraph.GetNeighborsCount(v);
  }

  bool is_directed;
  // These are initial nodes and edges count and not currenlty maintained after
  // updates
  SizeT nodes;
  SizeT edges;

  // The following code is for the old API (just for compiling)
  __device__ __host__ __forceinline__ void GetEdgeSrcDest(const SizeT &e,
                                                          VertexT &src,
                                                          VertexT &dest) const {
  }
  __device__ __host__ __forceinline__ SizeT
  GetSrcDestEdge(const VertexT &src, const VertexT &dest) const {
    return util::PreDefinedValues<SizeT>::InvalidValue;
  }
  __device__ __host__ __forceinline__ VertexT
  GetEdgeDest(const SizeT &e) const {
    return util::PreDefinedValues<VertexT>::InvalidValue;
  }
};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
