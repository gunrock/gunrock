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

#include <stdlib.h>

/**
 * @brief Vertex_id datatypes enumerators.
 * TODO: add more data types
 */
enum VertexIdType
{
    VTXID_INT,   //!< int type VertexId
};
enum SizeTType
{
    SIZET_UINT,   //!< unsigned int type SizeT
};
enum ValueType
{
    VALUE_INT,    //!< int    type Value
    VALUE_FLOAT,  //!< float  type Value
    //VALUE_DOUBLE, //!< double type Value
};

/**
 * @brief datatype configuration struct used to specify datatypes
 *
 */
struct GunrockDataType
{
    enum VertexIdType VTXID_TYPE; //!< VertexId datatype
    enum SizeTType    SIZET_TYPE; //!< SizeT    datatype
    enum ValueType    VALUE_TYPE; //!< Value    datatype
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
};

#ifdef __cplusplus
extern "C" {
#endif

void Test();

// topk algorithm
/*
void gunrock_topk(
  struct GunrockGraph *graph_out,
  void                *node_ids,
  void                *centrality_values,
  size_t              top_nodes,
  const struct GunrockGraph  *graph_in,
  struct GunrockDataType     data_type);
*/

// topk dispatch function
void topk_dispatch(
  struct GunrockGraph       *graph_out,
  void                      *node_ids,
  void                      *centrality_values,
  const struct GunrockGraph *graph_in,
  size_t                    top_nodes,
  struct GunrockDataType    data_type);

#ifdef __cplusplus
}
#endif

// TODO: Add other algorithms

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
