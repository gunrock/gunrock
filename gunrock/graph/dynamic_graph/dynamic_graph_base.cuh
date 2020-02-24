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

#include <gunrock/util/array_utils.cuh>
#include <gunrock/graph/csr.cuh>

#include <gunrock/graph/dynamic_graph/hash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/hash_graph_map.cuh>
#include <gunrock/graph/dynamic_graph/hash_graph_set.cuh>

namespace gunrock {
namespace graph {

template<
    typename _VertexT,
    typename _SizeT,
    typename _ValueT,
    GraphFlag FLAG>
struct DynamicGraphBase
{   
    public:
    static constexpr bool REQUIRE_SORTING = (FLAG /*& IS_SORTED*/) != 0;
    static constexpr bool REQUIRE_EDGES_VALUES = (FLAG & HAS_EDGE_VALUES) != 0;

    using VertexT = _VertexT;
    using SizeT = _SizeT;
    using ValueT = _ValueT;

    using HashGraphMapT = HashGraphMap<VertexT, SizeT, ValueT, FLAG>;
    using HashGraphSetT = HashGraphSet<VertexT, SizeT, ValueT, FLAG>;

    //Only one choice now
    using DynamicGraphT = typename std::conditional<REQUIRE_SORTING,
                                        typename std::conditional<REQUIRE_EDGES_VALUES, HashGraphMapT, HashGraphSetT>::type,
                                        typename std::conditional<REQUIRE_EDGES_VALUES, HashGraphMapT, HashGraphSetT>::type>::type;

    DynamicGraphT dynamicGraph;
};


} // namespace graph
} // namespace gunrock
