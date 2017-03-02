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
    GraphFlag FLAG   = GRAPH_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct Coo :
    public GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag>
{
    typedef GraphBase<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> BaseGraph;

    typedef typename util::VectorType<VertexT, 2>::Type
        EdgePairT;

    // whether the edges are edge_order
    EdgeOrder edge_order;

    // Source (.x) and Destination (.y) of edges
    util::Array1D<SizeT, EdgePairT,
        util::If_Val<FLAG & GRAPH_PINNED,
            util::PINNED, util::ARRAY_NONE>::Value,
        cudaHostRegisterFlag> edge_pairs;

    typedef util::Array1D<SizeT, ValueT,
        util::If_Val<FLAG & GRAPH_PINNED,
            util::PINNED, util::ARRAY_NONE>::Value,
        cudaHostRegisterFlag> Array_ValueT;

    // List of values attached to edges in the graph
    typename util::If<FLAG & HAS_EDGE_VALUES,
        Array_ValueT, util::NullArray<SizeT, ValueT,
            util::If_Val<FLAG & GRAPH_PINNED,
                util::PINNED, util::ARRAY_NONE>::Value,
            cudaHostRegisterFlag> >::Type edge_values;

    // List of values attached to nodes in the graph
    typename util::If<FLAG & HAS_NODE_VALUES,
        Array_ValueT, util::NullArray<SizeT, ValueT,
            util::If_Val<FLAG & GRAPH_PINNED,
                util::PINNED, util::ARRAY_NONE>::Value,
            cudaHostRegisterFlag> >::Type node_values;

    /**
     * @brief COO Constructor
     *
     * @param[in] pinned Use pinned memory for CSR data structure
     * (default: do not use pinned memory)
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
    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = edge_pairs    .Release()) return retval;
        if (retval = node_values   .Release()) return retval;
        if (retval = edge_values   .Release()) return retval;
        if (retval = BaseGraph    ::Release()) return retval;
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
            target = source.edge_pairs.setted | source.edge_pairs.allocated;

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

        // Sort
        if (FLAG & HAS_EDGE_VALUES)
        {
            if (new_order == BY_ROW_ASCENDING)
                retval = edge_pairs.Sort_by_Key(
                    edge_values, row_ascen_order,
                    this -> edges, 0, target, stream);
            else if (new_order == BY_ROW_DECENDING)
                retval = edge_pairs.Sort_by_Key(
                    edge_values, row_decen_order,
                    this -> edges, 0, target, stream);
            else if (new_order == BY_COLUMN_ASCENDING)
                retval = edge_pairs.Sort_by_Key(
                    edge_values, column_ascen_order,
                    this -> edges, 0, target, stream);
            else if (new_order == BY_ROW_DECENDING)
                retval = edge_pairs.Sort_by_Key(
                    edge_values, column_decen_order,
                    this -> edges, 0, target, stream);
            if (retval) return retval;
        } else { // no edge values
            if (new_order == BY_ROW_ASCENDING)
                retval = edge_pairs.Sort(
                    row_decen_order,
                    this -> edges, 0, target, stream);
            else if (new_order == BY_ROW_DECENDING)
                retval = edge_pairs.Sort(
                    row_decen_order,
                    this -> edges, 0, target, stream);
            else if (new_order == BY_COLUMN_ASCENDING)
                retval = edge_pairs.Sort(
                    column_ascen_order,
                    this -> edges, 0, target, stream);
            else if (new_order == BY_ROW_DECENDING)
                retval = edge_pairs.Sort(
                    column_decen_order,
                    this -> edges, 0, target, stream);
            if (retval) return retval;
        }

        return retval;
    }

}; // Coo

} // namespace graph

/**
 * @brief COO sparse format edge. (A COO graph is just a
 * list/array/vector of these.)
 *
 * @tparam VertexId Vertex identifiler type.
 * @tparam Value Attribute value type.
 *
 */
/*template<typename VertexId, typename Value>
struct Coo {
    VertexId row;
    VertexId col;
    Value val;

    Coo() {}
    Coo(VertexId row, VertexId col, Value val) : row(row), col(col), val(val) {}

    void Val(Value &value) {
        value = val;
    }
};*/


/*
 * @brief Coo data structure.
 *
 * @tparam VertexId Vertex identifier type.
 */
/*template<typename VertexId>
struct Coo<VertexId, util::NullType> {
    VertexId row;
    VertexId col;

    template <typename Value>
    Coo(VertexId row, VertexId col, Value val) : row(row), col(col) {}

    template <typename Value>
    void Val(Value &value) {}
};*/


/**
 * @brief Comparator for sorting COO sparse format edges first by row
 *
 * @tparam Coo COO Datatype
 *
 * @param[in] elem1 First element to compare
 * @param[in] elem2 Second element to compare
 * @returns true if first element comes before second element in (r,c)
 * order, otherwise false
 *
 * @see ColumnFirstTupleCompare
 */
/*template<typename Coo>
bool RowFirstTupleCompare (
    Coo elem1,
    Coo elem2) {
    if (elem1.row < elem2.row) {
        // Sort edges by source node
        return true;
    } else if ((elem1.row == elem2.row) && (elem1.col < elem2.col)) {
        // Sort edgelists as well for coherence
        return true;
    }

    return false;
}*/

/**
 * @brief Comparator for sorting COO sparse format edges first by column
 *
 * @tparam Coo COO Datatype
 *
 * @param[in] elem1 First element to compare
 * @param[in] elem2 Second element to compare
 * @returns true if first element comes before second element in (c,r)
 * order, otherwise false
 *
 * @see RowFirstTupleCompare
 */
/*template<typename Coo>
bool ColumnFirstTupleCompare (
    Coo elem1,
    Coo elem2) {
    if (elem1.col < elem2.col) {
        // Sort edges by source node
        return true;
    } else if ((elem1.col == elem2.col) && (elem1.row < elem2.row)) {
        // Sort edgelists as well for coherence
        return true;
    }

    return false;
}*/


} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
