// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gunrock.h
 *
 * @brief Main Library header file. Defines public interface.
 * The Gunrock public interface is a C-only interface to enable linking 
 * with code written in other languages. While the internals of Gunrock 
 * are not limited to C.
 *
 */

#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

using namespace gunrock::app::topk;
    
/**
 * @brief Vertex_id datatypes enumerators.
 * TODO: add more types
 */
enum VertexIdType
{
    VTXID_UINT,   //!< unsigned int type VertexId
    VTXID_ULLONG, //!< ussigned long long int type VertexId
};
enum SizeTType
{
    SIZET_UINT,   //!< unsigned int type SizeT
    SIZET_ULLONG, //!< unsigned long long int type SizeT
};
enum ValueType
{	
    VALUE_INT,    //!< int    type Value
    VALUE_FLOAT,  //!< float  type Value
    VALUE_DOUBLE, //!< double type Value 
};

/**
 * @brief datatype configuration struct used to specify datatypes
 * TODO: 
 */
struct GunrockDataType
{
    VertexIdType VTXID_TYPE; //!< VertexId datatype
    SizeTType    SIZET_TYPE; //!< SizeT    datatype
    ValueType    VALUE_TYPE; //!< Value    datatype
};

/**
 * @brief GunrockGraph struct as a standard graph interface
 */
struct GunrockGraph
{
    size_t  num_nodes;
    size_t  num_edges;
    void    *row_offsets;
    void    *col_indices;
    void    *col_offsets;
    void    *row_indices;
    void    *node_values;
    void    *edge_values;
}

// topk algorithm
void gunrock_topk(
                GunrockGraph        *g_out,
                void                *node_ids,
                void                *centrality_values,
                size_t              top_node,
                const GunrockGraph  *g_in,
                GunrockDataType     data_type);

// topk dispatch function
void topk_dispatch(
                GunrockGraph        *g_out,
                void                *node_ids,
                void                *centrality_values,
                size_t              top_node,
                const GunrockGraph  *g_in,
                GunrockDataType     data_type);

// TODO: Add other algorithms

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
