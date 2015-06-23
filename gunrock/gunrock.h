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
    size_t num_nodes;     // number of nodes in graph
    size_t num_edges;     // number of edges in graph
    void   *row_offsets;  // CSR row offsets
    void   *col_indices;  // CSR column indices
    void   *col_offsets;  // CSC column offsets
    void   *row_indices;  // CSC row indices
    void   *node_values;  // associated values per node
    void   *edge_values;  // associated values per edge
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
    bool  mark_pred;         // whether to mark predecessor or not
    bool  idempotence;       // whether or not to enable idempotent
    int   src_node;          // source vertex define where to start
    int   device;            // setting which device to use
    int   max_iter;          // maximum number of iterations allowed
    int   top_nodes;         // k value for top k / pagerank problem
    int   delta_factor;      // sssp delta-factor parameter
    float delta;             // pagerank specific value
    float error;             // pagerank specific value
    float queue_size;        // setting frontier queue size
    enum  SrcMode src_mode;  // source mode rand/largest_degree
};

#ifdef __cplusplus
extern "C" {
#endif

// breath-first search
void gunrock_bfs(
    struct       GRGraph *graph_o,
    const struct GRGraph *graph_i,
    struct       GRSetup  config,
    struct       GRTypes  data_t);

// betweenness centrality
void gunrock_bc(
    struct       GRGraph *graph_o,
    const struct GRGraph *graph_i,
    struct       GRSetup  config,
    struct       GRTypes  data_t);

// connected component
void gunrock_cc(
    struct       GRGraph *graph_o,
    unsigned int         *components,
    const struct GRGraph *graph_i,
    struct       GRSetup  config,
    struct       GRTypes  data_t);

// single-source shortest path
void gunrock_sssp(
    struct       GRGraph *graph_o,
    void                 *predecessor,
    const struct GRGraph *graph_i,
    struct       GRSetup  config,
    struct       GRTypes  data_t);

// page-rank
void gunrock_pagerank(
    struct       GRGraph *graph_o,
    void                 *node_ids,
    void                 *pagerank,
    const struct GRGraph *graph_i,
    struct       GRSetup  config,
    struct       GRTypes  data_t);

// degree centrality
void gunrock_topk(
    struct       GRGraph *graph_o,
    void                 *node_ids,
    void                 *in_degrees,
    void                 *out_degrees,
    const struct GRGraph *graph_i,
    struct       GRSetup  config,
    struct       GRTypes  data_t);

// minimum spanning tree
void gunrock_mst(
    struct       GRGraph *graph_o,
    const struct GRGraph *graph_i,
    struct       GRSetup config,
    struct       GRTypes data_t);

// TODO(ydwu): Add other primitives

#ifdef __cplusplus
}
#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
