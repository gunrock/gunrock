// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file gunrock.h
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
 * @brief VertexId data type enumerators.
 */
enum VertexIdType {
    VTXID_INT, //!< integer type
};

/**
 * @brief SizeT data type enumerators.
 */
enum SizeTType {
    SIZET_INT, //!< unsigned integer type
};

/**
 * @brief Value data type enumerators.
 */
enum ValueType {
    VALUE_INT,   //!< integer type
    VALUE_UINT,  //!< unsigned int type
    VALUE_FLOAT, //!< float type
};

/**
 * @brief data-type configuration used to specify data types
 */
struct GunrockDataType {
    enum VertexIdType VTXID_TYPE; //!< VertexId data-type
    enum SizeTType    SIZET_TYPE; //!< SizeT    data-type
    enum ValueType    VALUE_TYPE; //!< Value    data-type
};

/**
 * @brief GunrockGraph as a standard graph interface
 */
struct GunrockGraph {
    size_t num_nodes;    //!< number of nodes in graph
    size_t num_edges;    //!< number of edges in graph
    void   *row_offsets; //!< C.S.R. row offsets
    void   *col_indices; //!< C.S.R. column indices
    void   *col_offsets; //!< C.S.C. column offsets
    void   *row_indices; //!< C.S.C. row indices
    void   *node_values; //!< associated values per node
    void   *edge_values; //!< associated values per edge
};

/**
 * @brief Source Vertex Mode enumerators.
 */
enum SrcMode {
    manually,       //!< manually set up source node
    randomize,      //!< random generate source node
    largest_degree, //!< set to largest-degree node
};

/**
 * @brief arguments configuration used to specify arguments
 */
struct GunrockConfig {
    bool  mark_pred;        //!< whether to mark predecessor or not
    bool  idempotence;      //!< whether or not to enable idempotent
    int   src_node;         //!< source vertex define where to start
    int   device;           //!< setting which gpu device to use
    int   max_iter;         //!< maximum number of iterations allowed
    int   top_nodes;        //!< k value for topk / page_rank problem
    int   delta_factor;     //!< sssp delta-factor parameter
    float delta;            //!< pagerank specific value
    float error;            //!< pagerank specific value
    float queue_size;       //!< setting frontier queue size
    enum  SrcMode src_mode; //!< source mode rand/largest_degree
};

#ifdef __cplusplus
extern "C" {
#endif

// BFS Function Define
void gunrock_bfs_func(
    struct GunrockGraph       *graph_out,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// BC Function Define
void gunrock_bc_func(
    struct GunrockGraph       *graph_out,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// CC Function Define
void gunrock_cc_func(
    struct GunrockGraph       *graph_out,
    unsigned int              *components,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// SSSP Function Define
void gunrock_sssp_func(
    struct GunrockGraph       *graph_out,
    void                      *predecessor,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      congis,
    struct GunrockDataType    data_type);

// PR Function Define
void gunrock_pr_func(
    struct GunrockGraph       *graph_out,
    void                      *node_ids,
    void                      *page_rank,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// TopK Function Define
void gunrock_topk_func(
    struct GunrockGraph       *graph_out,
    void                      *node_ids,
    void                      *in_degrees,
    void                      *out_degrees,
    const struct GunrockGraph *graph_in,
    struct GunrockConfig      configs,
    struct GunrockDataType    data_type);

// TODO: Add other algorithms

#ifdef __cplusplus
}
#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
