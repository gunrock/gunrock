// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * coo.cuh
 *
 * @brief Coordinate Format (a.k.a. triplet format) Graph Data Structure
 */

#pragma once

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/vector_types.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/oprtr/1D_oprtr/sort.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief COO data structure which uses Coordinate
 * format to store a graph. It is a compressed way to present
 * the graph as a sparse matrix.
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 */
template<
    typename VertexT = int,
    typename SizeT   = VertexT,
    typename ValueT  = VertexT,
    GraphFlag _FLAG   = GRAPH_NONE | HAS_COO,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct Coo :
    public GraphBase<VertexT, SizeT, ValueT, _FLAG | HAS_COO, cudaHostRegisterFlag>
{
    static const GraphFlag FLAG = _FLAG | HAS_COO;
    static const util::ArrayFlag ARRAY_FLAG =
        util::If_Val<(FLAG & GRAPH_PINNED) != 0, (FLAG & ARRAY_RESERVE) | util::PINNED,
            FLAG & ARRAY_RESERVE>::Value;
    typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> BaseGraph;
    typedef Coo<VertexT, SizeT, ValueT, _FLAG, cudaHostRegisterFlag> CooT;

    typedef typename util::VectorType<VertexT, 2>::Type
        EdgePairT;

    // whether the edges are edge_order
    EdgeOrder edge_order;

    // Source (.x) and Destination (.y) of edges
    util::Array1D<SizeT, EdgePairT, ARRAY_FLAG,
        cudaHostRegisterFlag> edge_pairs;

    typedef util::Array1D<SizeT, ValueT, ARRAY_FLAG,
        cudaHostRegisterFlag> Array_ValueT;
    typedef util::NullArray<SizeT, ValueT, ARRAY_FLAG,
        cudaHostRegisterFlag> Array_NValueT;

    // List of values attached to edges in the graph
    typename util::If<(FLAG & HAS_EDGE_VALUES) != 0,
        Array_ValueT, Array_NValueT>::Type edge_values;

    // List of values attached to nodes in the graph
    typename util::If<(FLAG & HAS_NODE_VALUES) != 0,
        Array_ValueT, Array_NValueT>::Type node_values;

    /**
     * @brief COO Constructor
     */
    Coo() : BaseGraph()
    {
        edge_pairs    .SetName("edge_pairs");
        edge_values   .SetName("edge_values");
        node_values   .SetName("node_values");
        edge_order = UNORDERED;
    }

    /**
     * @brief COO destructor
     */
    ~Coo()
    {
        //Release();
    }

    /**
     * @brief Deallocates CSR graph
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = edge_pairs    .Release(target)) return retval;
        if (retval = node_values   .Release(target)) return retval;
        if (retval = edge_values   .Release(target)) return retval;
        if (retval = BaseGraph    ::Release(target)) return retval;
        return retval;
    }

    /**
     * @brief Allocate memory for COO graph.
     *
     * @param[in] nodes Number of nodes in COO-format graph
     * @param[in] edges Number of edges in COO-format graph
     */
    cudaError_t Allocate(SizeT nodes, SizeT edges,
        util::Location target = GRAPH_DEFAULT_TARGET)
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseGraph    ::Allocate(nodes, edges, target))
            return retval;
        if (retval = edge_pairs    .Allocate(edges      , target))
            return retval;
        if (retval = node_values   .Allocate(nodes      , target))
            return retval;
        if (retval = edge_values   .Allocate(edges      , target))
            return retval;
        return retval;
    }

    cudaError_t Display(
        std::string graph_prefix = "",
        SizeT edges_to_show = 40,
        bool  with_edge_values = true)
    {
        cudaError_t retval = cudaSuccess;
        if (edges_to_show > this -> edges)
            edges_to_show = this -> edges;
        util::PrintMsg(graph_prefix + "Graph containing " +
            std::to_string(this -> nodes) + " vertices, " +
            std::to_string(this -> edges) + " edges, in COO format, ordered " + EdgeOrder_to_string(edge_order)
            + ". First " + std::to_string(edges_to_show) +
            " edges :");
        for (SizeT e=0; e < edges_to_show; e++)
            util::PrintMsg("e " + std::to_string(e) +
                " : " + std::to_string(edge_pairs[e].x) +
                " -> " + std::to_string(edge_pairs[e].y) +
                (((FLAG & HAS_EDGE_VALUES) && (with_edge_values))? (" (" + std::to_string(edge_values[e]) + ")") : ""));
        return retval;
    }

    template <
        typename VertexT_in, typename SizeT_in,
        typename ValueT_in, GraphFlag FLAG_in,
        unsigned int cudaHostRegisterFlag_in>
    cudaError_t FromCoo(
        Coo<VertexT_in, SizeT_in, ValueT_in, FLAG_in,
            cudaHostRegisterFlag_in> &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;
        if (target == util::LOCATION_DEFAULT)
            target = source.edge_pairs.GetSetted() | source.edge_pairs.GetAllocated();

        this -> edge_order = source.edge_order;
        if (retval = BaseGraph::Set(source))
            return retval;

        if (retval = Allocate(source.nodes, source.edges, target))
            return retval;

        if (retval = edge_pairs   .Set(source.edge_pairs,
            this -> edges, target, stream))
            return retval;

        if (retval = edge_values   .Set(source.edge_values,
            this -> edges, target, stream))
            return retval;

        if (retval = node_values   .Set(source.node_values,
            this -> nodes, target, stream))
            return retval;

        return retval;
    }

    template <typename GraphT>
    cudaError_t FromCsr(
        GraphT &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0)
    {
        typedef typename GraphT::CsrT CsrT;
        cudaError_t retval = cudaSuccess;
        if (target == util::LOCATION_DEFAULT)
            target = source.CsrT::row_offsets.GetSetted() | source.CsrT::row_offsets.GetAllocated();

        //if (retval = BaseGraph::Set(source))
        //    return retval;
        this -> nodes = source.CsrT::nodes;
        this -> nodes = source.CsrT::edges;
        this -> directed = source.CsrT::directed;
        this -> edge_order = UNORDERED;

        if (retval = Allocate(source.CsrT::nodes, source.CsrT::edges, target))
            return retval;

        if (retval = source.row_offsets.ForAll(
            edge_pairs, source.column_indices,
            [] __host__ __device__ (
                typename CsrT::SizeT *row_offsets,
                EdgePairT *edge_pairs,
                typename CsrT::VertexT *column_indices,
                const VertexT &row){
                    SizeT e_end = row_offsets[row+1];
                    for (SizeT e = row_offsets[row]; e < e_end; e++)
                    {
                        edge_pairs[e].x = row;
                        edge_pairs[e].y = column_indices[e];
                    }
                }, this -> nodes, target, stream))
            return retval;

        if (retval = edge_values   .Set(source.CsrT::edge_values,
            this -> edges, target, stream))
            return retval;

        if (retval = node_values   .Set(source.CsrT::node_values,
            this -> nodes, target, stream))
            return retval;
        return retval;
    }

    template <typename GraphT>
    cudaError_t FromCsc(
        GraphT &source,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0)
    {
        typedef typename GraphT::CscT CscT;

        cudaError_t retval = cudaSuccess;
        if (target == util::LOCATION_DEFAULT)
            target = source.edge_pairs.GetSetted() | source.edge_pairs.GetAllocated();

        //if (retval = BaseGraph::template Set<typename CscT::CscT>((typename CscT::CscT)source))
        //    return retval;
        this -> nodes = source.CscT::nodes;
        this -> edges = source.CscT::edges;
        this -> directed = source.CscT::directed;
        this -> edge_order = UNORDERED;

        if (retval = Allocate(source.CscT::nodes, source.CscT::edges, target))
            return retval;

        //util::PrintMsg("1");
        //for (SizeT v = 0; v<this -> nodes; v++)
        //    printf("O[%d] = %d\t", v, source.column_offsets[v]);
        //printf("\n");
        //fflush(stdout);

        if (retval = source.column_offsets.ForAll(
            edge_pairs, source.row_indices,
            [] __host__ __device__ (
                typename CscT::SizeT *column_offsets,
                EdgePairT *edge_pairs,
                typename CscT::VertexT *row_indices,
                const VertexT &column){
                    SizeT e_end = column_offsets[column+1];
                    for (SizeT e = column_offsets[column]; e < e_end; e++)
                    {
                        edge_pairs[e].x = row_indices[e];
                        edge_pairs[e].y = column;
                    }
                }, this -> nodes, target, stream))
            return retval;

        //util::PrintMsg("2");
        if (retval = edge_values   .Set(source.CscT::edge_values,
            this -> edges, target, stream))
            return retval;

        //util::PrintMsg("3");
        if (retval = node_values   .Set(source.CscT::node_values,
            this -> nodes, target, stream))
            return retval;
        //util::PrintMsg("4");
        return retval;
    }

    cudaError_t Order(EdgeOrder new_order = BY_ROW_ASCENDING,
        util::Location target = util::LOCATION_DEFAULT,
        cudaStream_t stream = 0)
    {
        cudaError_t retval = cudaSuccess;

        if (new_order == edge_order)
            return retval;

        if (target == util::LOCATION_DEFAULT)
            target = edge_pairs.GetSetted() | edge_pairs.GetAllocated();

        auto row_ascen_order = []__host__ __device__
            (const EdgePairT &e1, const EdgePairT &e2){
            if (e1.x > e2.x) return false;
            if (e1.x < e2.x) return true;
            if (e1.y > e2.y) return false;
            return true;
        };

        auto row_decen_order = []__host__ __device__
            (const EdgePairT &e1, const EdgePairT &e2){
            if (e1.x < e2.x) return false;
            if (e1.x > e2.x) return true;
            if (e1.y < e2.y) return false;
            return true;
        };

        auto column_ascen_order = []__host__ __device__
            (const EdgePairT &e1, const EdgePairT &e2){
            if (e1.y > e2.y) return false;
            if (e1.y < e2.y) return true;
            if (e1.x > e2.x) return false;
            return true;
        };

        auto column_decen_order = []__host__ __device__
            (const EdgePairT &e1, const EdgePairT &e2){
            if (e1.y < e2.y) return false;
            if (e1.y > e2.y) return true;
            if (e1.x < e2.x) return false;
            return true;
        };

        //util::PrintMsg("Before Sorting");
        //Display();
        // Sort
        if (FLAG & HAS_EDGE_VALUES)
        {
            switch (new_order)
            {
            case BY_ROW_ASCENDING:
                retval = edge_pairs.Sort_by_Key(
                    edge_values, row_ascen_order,
                    this -> edges, 0, target, stream);
                break;
            case BY_ROW_DECENDING:
                retval = edge_pairs.Sort_by_Key(
                    edge_values, row_decen_order,
                    this -> edges, 0, target, stream);
                break;
            case BY_COLUMN_ASCENDING:
                retval = edge_pairs.Sort_by_Key(
                    edge_values, column_ascen_order,
                    this -> edges, 0, target, stream);
                break;
            case BY_COLUMN_DECENDING:
                retval = edge_pairs.Sort_by_Key(
                    edge_values, column_decen_order,
                    this -> edges, 0, target, stream);
                break;
            case UNORDERED:
                break;
            }
            if (retval) return retval;
        } else { // no edge values
            switch (new_order)
            {
            case BY_ROW_ASCENDING:
                retval = edge_pairs.Sort(
                    row_ascen_order,
                    this -> edges, 0, target, stream);
                break;
            case BY_ROW_DECENDING:
                retval = edge_pairs.Sort(
                    row_decen_order,
                    this -> edges, 0, target, stream);
                break;
            case BY_COLUMN_ASCENDING:
                retval = edge_pairs.Sort(
                    column_ascen_order,
                    this -> edges, 0, target, stream);
                break;
            case BY_COLUMN_DECENDING:
                retval = edge_pairs.Sort(
                    column_decen_order,
                    this -> edges, 0, target, stream);
                break;
            case UNORDERED:
                break;
            }
            if (retval) return retval;
        }

        edge_order = new_order;
        //util::PrintMsg("After sorting");
        //Display();
        return retval;
    }

}; // Coo

} // namespace graph
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
