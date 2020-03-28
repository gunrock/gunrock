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

#include <gunrock/graph/dynamic_graph/slabhash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_map.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_set.cuh>

namespace gunrock {
namespace graph {

template<
    typename VertexT,
    typename SizeT,
    typename ValueT,
    GraphFlag FLAG>
struct DynamicGraphBase
{   
    public:
    static constexpr bool REQUIRE_SORTING = (FLAG /*& IS_SORTED*/) != 0;
    static constexpr bool REQUIRE_EDGES_VALUES = (FLAG & HAS_EDGE_VALUES) != 0;

    using HashGraphMapT = HashGraphMap<VertexT, SizeT, ValueT, REQUIRE_EDGES_VALUES>;
    using HashGraphSetT = HashGraphSet<VertexT, SizeT, ValueT, REQUIRE_EDGES_VALUES>;

    //Only one choice now
    using DynamicGraphT = typename std::conditional<REQUIRE_SORTING,
                                        typename std::conditional<REQUIRE_EDGES_VALUES, HashGraphMapT, HashGraphSetT>::type,
                                        typename std::conditional<REQUIRE_EDGES_VALUES, HashGraphMapT, HashGraphSetT>::type>::type;

    DynamicGraphT dynamicGraph;


    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        dynamicGraph.Release();
        return cudaSuccess;
    }

    bool is_directed;
};


} // namespace graph
} // namespace gunrock
