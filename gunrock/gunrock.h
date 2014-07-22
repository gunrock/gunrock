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
#include <stdbool.h>

/**
 * @brief Vertex_id datatypes enumerators.
 * TODO: add more data types
 */
enum VertexIdType
{
    VTXID_INT,   //!< integer type
};
enum SizeTType
{
    SIZET_UINT,   //!< unsigned integer type
};
enum ValueType
{
    VALUE_INT,    //!< integer type
    VALUE_FLOAT,  //!< float type
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
    size_t  num_nodes;    //!< number of nodes in graph
    size_t  num_edges;    //!< number of edges in graph
    void    *row_offsets; //!< C.S.R. row offsets
    void    *col_indices; //!< C.S.R. column indices
    void    *col_offsets; //!< C.S.C. column offsets
    void    *row_indices; //!< C.S.C. row indices
    void    *node_values; //!< associated values per node
    void    *edge_values; //!< associated values per edge
};

/**
 * @brief arguments configuration struct used to specify arguments
 */
struct GunrockConfig
{
    bool  undirected;  //!< whether the graph is undirected or not
    bool  mark_pred;   //!< whether to mark predecessor or not
    bool  idempotence; //!< whether or not to enable idempotence
    int   source;      //!< source vertex define where to start
    int   device;      //!< setting which gpu device to use
    int   max_iter;    //!< maximum mumber of iterations allowed
    int   top_nodes;   //!< k value or top nodes for topk problem
    float alpha;       //!< betweeness centrality specific value
    float beta;        //!< betweeness centrality specific value
    float delta;       //!< page rank specific value
    float error;       //!< page rank specific value
    float queue_size;  //!< setting frontier queue size
    bool  help;        //!< whether to print the help infomation
};

#ifdef __cplusplus
extern "C" {
#endif

// BFS Function Define
void gunrock_bfs(
    struct GunrockGraph       *graph_out,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// BC Function Define
void gunrock_bc(
    struct GunrockGraph       *graph_out,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// CC Function Define
void gunrock_cc(
    struct GunrockGraph       *graph_out,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// SSSP Function Define

// PR Function Define

// TODO: Add other algorithms

// TopK Implementation
void gunrock_topk(
    struct GunrockGraph       *graph_out,
    void                      *node_ids,
    void                      *centrality_values,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      topk_config,
    struct GunrockDataType    data_type);

#ifdef __cplusplus
}
#endif



// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: