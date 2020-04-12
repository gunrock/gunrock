// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * dynamic_graph.cuh
 *
 * @brief DYN (Dynamic) Graph Data Structure
 */
#pragma once

#include <gunrock/graph/dynamic_graph/dynamic_graph_base.cuh>
#include <gunrock/graph/graph_base.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief Dynamic graph data structure which uses
 * a per-vertex data structure based on the graph flags.
 * Specialized for weighted or unweighted graphs
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename _VertexT = int, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT, GraphFlag _FLAG = GRAPH_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault,
          bool VALID = true, bool HAS_VALUES = ((_FLAG & HAS_EDGE_VALUES) != 0)>
struct Dyn : DynamicGraphBase<_VertexT, _SizeT, _ValueT, _FLAG> {};

}  // namespace graph
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
