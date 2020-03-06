// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * dyn.cuh
 *
 * @brief DYN (Dynamic) Graph Data Structure
 */
#pragma once

#include <gunrock/graph/dynamic_graph/dynamic_graph_base.cuh>
#include <gunrock/graph/graph_base.cuh>


namespace gunrock {
namespace graph {

template<
    typename _VertexT = int,
    typename _SizeT   = _VertexT,
    typename _ValueT  = _VertexT,
    GraphFlag _FLAG   = GRAPH_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault,
    bool VALID = true,
    bool HAS_VALUES = ((_FLAG & HAS_EDGE_VALUES) != 0)>
struct Dyn : public DynamicGraphBase<_VertexT, _SizeT, _ValueT, _FLAG> {};


} // namespace graph
} // namespace gunrock
