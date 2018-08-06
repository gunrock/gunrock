// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file gunrock.h
 *
 * @brief Main library header file. Defines public C interface.
 * The Gunrock public interface is a C-only interface to enable linking
 * with code written in other languages. While the internals of Gunrock
 * are not limited to C.
 */

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief VertexId data type enumerators.
 */
enum VtxIdType
{
    VTXID_INT,  // Integer
};

/**
 * @brief SizeT data type enumerators.
 */
enum SizeTType
{
    SIZET_INT,  // Unsigned integer
};

/**
 * @brief Value data type enumerators.
 */
enum ValueType
{
    VALUE_INT,    // Integer
    VALUE_UINT,   // Unsigned integer
    VALUE_FLOAT,  // Float
};

/**
 * @brief Data type configuration used to specify data types.
 */
struct GRTypes
{
    enum VtxIdType VTXID_TYPE;  // VertexId data type
    enum SizeTType SIZET_TYPE;  // SizeT data type
    enum ValueType VALUE_TYPE;  // Value data type
};

/**
 * @brief GunrockGraph as a standard graph interface.
 */
struct GRGraph
{
    size_t  num_nodes;  // Number of nodes in graph
    size_t  num_edges;  // Number of edges in graph
    void *row_offsets;  // CSR row offsets
    void *col_indices;  // CSR column indices
    void *col_offsets;  // CSC column offsets
    void *row_indices;  // CSC row indices
    void *edge_values;  // Associated values per edge

    void *node_value1;  // Associated values per node
    void *edge_value1;  // Associated values per edge
    void *node_value2;  // Associated values per node
    void *edge_value2;  // Associated values per edge
    void *aggregation;  // Global reduced aggregation
};

/**
 * @brief Source Vertex Mode enumerators.
 */
enum SrcMode
{
    manually,        // Manually set up source node
    randomize,       // Random generate source node
    largest_degree,  // Largest-degree node as source
};

/**
 * @brief arguments configuration used to specify arguments.
 */
struct GRSetup
{
    bool               quiet;  // Whether to print out to STDOUT
    bool   mark_predecessors;  // Whether to mark predecessor or not
    bool  enable_idempotence;  // Whether or not to enable idempotent
    int*       source_vertex;  // Source nodes define where to start
    int            num_iters;  // Number of BFS runs (currently only support BFS)
    int         delta_factor;  // SSSP delta-factor parameter
    int*         device_list;  // Setting which device(s) to use
    unsigned int num_devices;  // Number of devices for computation
    unsigned int   max_iters;  // Maximum number of iterations allowed
    unsigned int   top_nodes;  // K value for top k / PageRank problem
    float     pagerank_delta;  // PageRank specific value
    float     pagerank_error;  // PageRank specific value
    bool pagerank_normalized;  // PageRank specific flag
    float   max_queue_sizing;  // Setting frontier queue size
    char* traversal_mode;  // Traversal mode: 0 for LB, 1 TWC
    enum SrcMode source_mode;  // Source mode rand/largest_degree
};

/**
 * @brief Initialization function for GRSetup.
 *
 * @param[out] num_iters Rounds of graph primitive to run.
 * @param[in]  source Pointer to source nodes array for each round.
 *
 * \return Initialized configurations object.
 */
// Proper way to check for C99
#if __STDC_VERSION__ >= 199901L
// http://clang.llvm.org/compatibility.html#inline
// Link mentions is an issue with C99, not a clang specific issue
static
#endif
inline struct GRSetup* InitSetup(int num_iters, int* source)
{
    struct GRSetup *configurations = (struct GRSetup*)malloc(sizeof(struct GRSetup));
    configurations -> quiet = true;
    configurations -> mark_predecessors = true;
    configurations -> enable_idempotence = false;
    int* sources = (int*)malloc(sizeof(int)*num_iters);
    int i;
    if (source == NULL)
    {
        for (i = 0; i < num_iters; ++i) sources[i] = 0;
    } else
    {
        for (i = 0; i < num_iters; ++i) sources[i] = source[i];
    }
    configurations -> source_vertex = sources;
    configurations -> delta_factor = 32;
    configurations -> num_devices = 1;
    configurations -> max_iters = 50;
    configurations -> num_iters = num_iters;
    configurations -> top_nodes = 10;
    configurations -> pagerank_delta = 0.85f;
    configurations -> pagerank_error = 0.01f;
    configurations -> pagerank_normalized = false;
    configurations -> max_queue_sizing = 1.0;
    configurations -> traversal_mode = (char*)malloc(sizeof(char) * 3);
    strcpy(configurations -> traversal_mode, "LB");
    configurations -> traversal_mode[2] = '\0';
    configurations -> source_mode = manually;
    int* gpu_idx = (int*)malloc(sizeof(int)); gpu_idx[0] = 0;
    configurations -> device_list = gpu_idx;
    return configurations;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Breath-first search public interface.
 *
 * @param[out] grapho Output data structure contains results.
 * @param[in]  graphi Input data structure contains graph.
 * @param[in]  config Primitive-specific configurations.
 * @param[in]  data_t Primitive-specific data type setting.
 *
 * \return Elapsed run time in milliseconds
 */
float gunrock_bfs(
    struct GRGraph*       grapho,   // Output graph / results
    const struct GRGraph* graphi,   // Input graph structure
    const struct GRSetup* config,   // Flag configurations
    const struct GRTypes  data_t);  // Data type Configurations

/*
 * @brief Simple interface take in CSR arrays as input
 *
 * @param[out] bfs_label            Return BFS label (depth) per nodes
 * @param[out] bfs_label            Return the predecessor per nodes
 * @param[in]  num_nodes            Number of nodes of the input graph
 * @param[in]  num_edges            Number of edges of the input graph
 * @param[in]  row_offsets          CSR-formatted graph input row offsets
 * @param[in]  col_indices          CSR-formatted graph input column indices
 * @param[in]  num_iters            Number of BFS runs. Note if num_iters > 1, the bfs_lbel will only store the results from the last run
 * @param[in]  source               Sources to begin traverse
 * @param[in]  source_mode          Enumerator of source mode: manually, randomize, largest_degree
 * @param[in]  mark_predecessors    If the flag is set, mark predecessors instead of bfs label
 * @param[in]  enable_idempotence   If the flag is set, use optimizations that allow idempotence operation (will usually bring better performance)
 */
float bfs(
    int*       bfs_label,
    int*       bfs_pred,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  num_iters,
    int* source,
    enum SrcMode source_mode,
    const bool mark_predecessors,
    const bool enable_idempotence);

/**
 * @brief Betweenness centrality public interface.
 *
 * @param[out] grapho Output data structure contains results.
 * @param[in]  graphi Input data structure contains graph.
 * @param[in]  config Primitive-specific configurations.
 * @param[in]  data_t Primitive-specific data type setting.
 */
void gunrock_bc(
    struct GRGraph*       grapho,   // Output graph / results
    const struct GRGraph* graphi,   // Input graph structure
    const struct GRSetup* config,   // Flag configurations
    const struct GRTypes  data_t);  // Data type Configurations

/**
 * @brief Betweenness centrality simple public interface.
 *
 * @param[out] bc_scores Return betweenness centralities.
 * @param[in] num_nodes Input graph number of nodes.
 * @param[in] num_edges Input graph number of edges.
 * @param[in] row_offsets Input graph row_offsets.
 * @param[in] col_indices Input graph col_indices.
 * @param[in] source Source node to start.
 */
void bc(
    float*     bc_scores,    // Return centrality score per node
    const int  num_nodes,    // Input graph number of nodes
    const int  num_edges,    // Input graph number of edges
    const int* row_offsets,  // Input graph row_offsets
    const int* col_indices,  // Input graph col_indices
    const int  source);      // Source vertex to start

/**
 * @brief Connected component public interface.
 *
 * @param[out] grapho Output data structure contains results.
 * @param[in]  graphi Input data structure contains graph.
 * @param[in]  config Primitive-specific configurations.
 * @param[in]  data_t Primitive-specific data type setting.
 */
void gunrock_cc(
    struct GRGraph*       grapho,   // Output graph / results
    const struct GRGraph* graphi,   // Input graph structure
    const struct GRSetup* config,   // Flag configurations
    const struct GRTypes  data_t);  // Data type Configurations

/**
 * @brief Connected component simple public interface.
 *
 * @param[out] component Return per-node component IDs.
 * @param[in] num_nodes Input graph number of nodes.
 * @param[in] num_edges Input graph number of edges.
 * @param[in] row_offsets Input graph row_offsets.
 * @param[in] col_indices Input graph col_indices.

 *\return int number of connected components in the graph.
 */
int cc(
    int*       component,     // Return component IDs per node
    const int  num_nodes,     // Input graph number of nodes
    const int  num_edges,     // Input graph number of edges
    const int* row_offsets,   // Input graph row_offsets
    const int* col_indices);  // Input graph col_indices

/**
 * @brief Single-source shortest path public interface.
 *
 * @param[out] grapho Output data structure contains results.
 * @param[in]  graphi Input data structure contains graph.
 * @param[in]  config Primitive-specific configurations.
 * @param[in]  data_t Primitive-specific data type setting.
 *
 * \return Elapsed run time in milliseconds
 */
float gunrock_sssp(
    struct GRGraph*       grapho,   // Output graph / results
    const struct GRGraph* graphi,   // Input graph structure
    const struct GRSetup* config,   // Flag configurations
    const struct GRTypes  data_t);  // Data type Configurations

/**
 * @brief Single-source shortest path simple public interface.
 *
 * @param[out] distances Return shortest distances.
 * @param[out] preds Return predecessor of each node
 * @param[in] num_nodes Input graph number of nodes.
 * @param[in] num_edges Input graph number of edges.
 * @param[in] row_offsets Input graph row_offsets.
 * @param[in] col_indices Input graph col_indices.
 * @param[in] edge_values Input graph edge weight.
 * @param[in] num_iters How many rounds of SSSP do we want to run.
 * @param[in] source Source node to start.
 * @param[in] mark_preds Whether to mark the predecessors.
 *
 * \return Elapsed run time in milliseconds
 */
float sssp(
    unsigned int*       distances,    // Return shortest distances
    int*                preds,
    const int           num_nodes,    // Input graph number of nodes
    const int           num_edges,    // Input graph number of edges
    const int*          row_offsets,  // Input graph row_offsets
    const int*          col_indices,  // Input graph col_indices
    const unsigned int* edge_values,  // Input graph edge weight
    const int           num_iters,
    int*                source,
    const bool          mark_preds);

/**
 * @brief PageRank public interface.
 *
 * @param[out] grapho Output data structure contains results.
 * @param[in]  graphi Input data structure contains graph.
 * @param[in]  config Primitive-specific configurations.
 * @param[in]  data_t Primitive-specific data type setting.
 */
/*void gunrock_pagerank(
    struct GRGraph*       grapho,   // Output graph / results
    const struct GRGraph* graphi,   // Input graph structure
    const struct GRSetup* config,   // Flag configurations
    const struct GRTypes  data_t);  // Data type Configurations
*/
/**
 * @brief PageRank simple public interface.
 *
 * @param[out] node_ids Return top-ranked vertex IDs.
 * @param[out] pagerank Return top-ranked PageRank scores.
 * @param[in] num_nodes Input graph number of nodes.
 * @param[in] num_edges Input graph number of edges.
 * @param[in] row_offsets Input graph row_offsets.
 * @param[in] col_indices Input graph col_indices.
 * @param[in] normalized Whether to perform a normalized PageRank
 */
double pagerank(
    const int  num_nodes,     // Input graph number of nodes
    const int  num_edges,     // Input graph number of edges
    const int* row_offsets,   // Input graph row_offsets
    const int* col_indices,   // Input graph col_indices
    bool       normalize,   // normalized pagerank flag
    int*       node_ids,
    float*     ranks);

// TODO Add other primitives

#ifdef __cplusplus
}
#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
