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
enum GraphFlag: unsigned int
{
    GRAPH_NONE      = 0x00,
    HAS_EDGE_VALUES = 0x01,
    HAS_NODE_VALUES = 0x02,
    HAS_CSR         = 0x10,
    HAS_CSC         = 0x20,
    HAS_COO         = 0x30,

    GRAPH_PINNED    = 0x100,
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
    const GraphFlag FLAG = _FLAG;
    const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;

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

    cudaError_t Release()
    {
        nodes = 0;
        edges = 0;
        directed = true;
        return cudaSuccess;
    }

    cudaError_t Allocate(SizeT nodes, SizeT edges,
        util::Location target = GRAPH_DEFAULT_TARGET)
    {
        return cudaSuccess;
    }

    template <
        typename VertexT_in, typename SizeT_in,
        typename ValueT_in, GraphFlag FLAG_in,
        unsigned int cudaHostRegisterFlag_in>
    cudaError_t Set(
        GraphBase<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
            cudaHostRegisterFlag_in> &source,
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
