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
 */

#include <stdlib.h>
#include <stdbool.h>

/**
 * @brief VertexId data type enumerators.
 */
enum VtxIdType {
    VTXID_INT,  // integer type
};

/**
 * @brief SizeT data type enumerators.
 */
enum SizeTType {
    SIZET_INT,  // unsigned integer type
};

/**
 * @brief Value data type enumerators.
 */
enum ValueType {
    VALUE_INT,    // integer type
    VALUE_UINT,   // unsigned int type
    VALUE_FLOAT,  // float type
};

/**
 * @brief data-type configuration used to specify data types
 */
struct GRTypes {
    enum VtxIdType VTXID_TYPE;  // VertexId data type
    enum SizeTType SIZET_TYPE;  // SizeT data type
    enum ValueType VALUE_TYPE;  // Value data type
};

/**
 * @brief GunrockGraph as a standard graph interface
 */
struct GRGraph {
    size_t  num_nodes;  // number of nodes in graph
    size_t  num_edges;  // number of edges in graph
    void *row_offsets;  // CSR row offsets
    void *col_indices;  // CSR column indices
    void *col_offsets;  // CSC column offsets
    void *row_indices;  // CSC row indices
    void *node_value1;  // associated values per node
    void *edge_value1;  // associated values per edge
    void *node_value2;  // associated values per node
    void *edge_value2;  // associated values per edge
    //void  aggregation;  // global reduced aggregation
};

/**
 * @brief Source Vertex Mode enumerators.
 */
enum SrcMode {
    manually,        // manually set up source node
    randomize,       // random generate source node
    largest_degree,  // set to largest-degree node
};

/**
 * @brief arguments configuration used to specify arguments
 */
struct GRSetup {
    bool   mark_predecessors;  // whether to mark predecessor or not
    bool  enable_idempotence;  // whether or not to enable idempotent
    int        source_vertex;  // source vertex define where to start
    int         delta_factor;  // sssp delta-factor parameter
    int*         device_list;  // setting which device(s) to use
    unsigned int num_devices;  // number of devices for computation
    unsigned int   max_iters;  // maximum number of iterations allowed
    unsigned int   top_nodes;  // k value for top k / pagerank problem
    float     pagerank_delta;  // pagerank specific value
    float     pagerank_error;  // pagerank specific value
    float   max_queue_sizing;  // setting frontier queue size
    enum SrcMode source_mode;  // source mode rand/largest_degree
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief R-MAT graph generator
 */
void rmat_graph(
    int          row_offsets[],
    int          col_indices[],
    unsigned int edge_values[],
    int           number_nodes,
    int           number_edges,
    bool         is_undirected,
    float     rmat_parameter_a,   // default for rmat 0.57
    float     rmat_parameter_b,   // default for rmat 0.19
    float     rmat_parameter_c);  // default for rmat 0.19

/**
 * breath-first search
 */
void gunrock_bfs(
    struct GRGraph*       graph_o,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

void bfs(
    int*       bfs_label,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  source);

/**
 * betweenness centrality
 */
void gunrock_bc(
    struct GRGraph*       graph_o,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

void bc(
    float*     bc_scores,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  source);

/**
 * connected component
 */
void gunrock_cc(
    struct GRGraph*       graph_o,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

int cc(
    int*       component,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices);

/**
 * single-source shortest path
 */
void gunrock_sssp(
    struct GRGraph*       graph_o,
    void*                 predecessor,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

void sssp(
    unsigned int*       distances,
    const int           num_nodes,
    const int           num_edges,
    const int*          row_offsets,
    const int*          col_indices,
    const unsigned int* edge_values,
    const int           source);

// pagerank
void gunrock_pagerank(
    struct GRGraph*       graph_o,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

void pagerank(
    int*       node_ids,
    float*     pagerank,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices);

// degree centrality
void gunrock_topk(
    struct  GRGraph*      graph_o,
    void*                 node_ids,
    void*                 in_degrees,
    void*                 out_degrees,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

// minimum spanning tree
void gunrock_mst(
    struct GRGraph*       graph_o,
    const struct GRGraph* graph_i,
    const struct GRSetup  config,
    const struct GRTypes  data_t);

void mst(
    bool*      edge_mask,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices);

// TODO(ydwu): Add other primitives

#ifdef __cplusplus
}
#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
