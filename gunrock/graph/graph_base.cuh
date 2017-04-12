// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * graph_base.cuh
 *
 * @brief Base Graph Data Structure
 */

#pragma once

namespace gunrock {
namespace graph {

#define ENABLE_GRAPH_DEBUG

/**
 * @brief Predefined flags for graph types
 */
using GraphFlag = unsigned int;
enum {
    ARRAY_RESERVE   = 0x000F,

    GRAPH_NONE      = 0x0000,
    HAS_EDGE_VALUES = 0x0010,
    HAS_NODE_VALUES = 0x0020,
    HAS_CSR         = 0x0100,
    HAS_CSC         = 0x0200,
    HAS_COO         = 0x0400,
    HAS_GP          = 0x0800,

    GRAPH_PINNED    = 0x1000,
};

static const util::Location GRAPH_DEFAULT_TARGET = util::DEVICE;

/**
 * @brief Enum to show how the edges are ordered
 */
enum EdgeOrder
{
    BY_ROW_ASCENDING,
    BY_ROW_DECENDING,
    BY_COLUMN_ASCENDING,
    BY_COLUMN_DECENDING,
    UNORDERED,
};

std::string EdgeOrder_to_string(EdgeOrder order)
{
    switch (order)
    {
    case BY_ROW_ASCENDING: return "by row ascending";
    case BY_ROW_DECENDING: return "by row decending";
    case BY_COLUMN_ASCENDING: return "by column ascending";
    case BY_COLUMN_DECENDING: return "by column decending";
    case UNORDERED: return "unordered";
    }
    return "unspecified";
}

/**
 * @brief GraphBase data structure to store basic info about a graph.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam GraphFlag graph flag
 */
template <typename _VertexT, typename _SizeT,
    typename _ValueT, GraphFlag _FLAG,
    unsigned int _cudaHostRegisterFlag = cudaHostRegisterDefault>
struct GraphBase
{
    typedef _VertexT VertexT;
    typedef _SizeT   SizeT;
    typedef _ValueT  ValueT;
    static const GraphFlag FLAG = _FLAG;
    static const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;

    SizeT nodes;   // Number of nodes in the graph
    SizeT edges;   // Number of edges in the graph
    bool  directed; // Whether the graph is directed

    GraphBase() :
        nodes (0),
        edges (0),
        directed(true)
    {}

    ~GraphBase()
    {
        //Release();
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        nodes = 0;
        edges = 0;
        directed = true;
        return cudaSuccess;
    }

    cudaError_t Allocate(SizeT nodes, SizeT edges,
        util::Location target = GRAPH_DEFAULT_TARGET)
    {
        this -> nodes = nodes;
        this -> edges = edges;
        return cudaSuccess;
    }

    template <typename GraphT_in>
    cudaError_t Set(
        GraphT_in &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;
        this -> nodes = source.nodes;
        this -> edges = source.edges;
        this -> directed = source.directed;
        return retval;
    }
};

} // namespace graph
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
