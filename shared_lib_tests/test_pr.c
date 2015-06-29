/**
 * @brief PR test for shared library
 * @file test_pr.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    // define data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;    // integer type vertex_ids
    data_t.SIZET_TYPE = SIZET_INT;    // integer type graph size
    data_t.VALUE_TYPE = VALUE_FLOAT;  // float type value for pr

    // pr configurations (optional)
    struct GRSetup config;
    config.device    =     0;  // use device 0
    config.delta     = 0.85f;  // default delta value
    config.error     = 0.01f;  // default error threshold
    config.max_iter  =    20;  // maximum number of iterations
    config.top_nodes =    10;  // number of top nodes
    config.src_node  =     0;  // source node to begin page rank
    config.src_mode  = manually;  // set source node manually

    // define graph (undirected graph)
    size_t num_nodes    = 7;
    size_t num_edges    = 15;
    int row_offsets[8]  = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};

    // build graph as input
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    // malloc output graph
    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    int   *node_ids  = (int*)malloc(sizeof(int) * config.top_nodes);
    float *pagerank = (float*)malloc(sizeof(float) * config.top_nodes);

    // run pr calculations
    gunrock_pagerank(graph_o, node_ids, pagerank, graph_i, config, data_t);

    // test print
    int i;
    printf("Demo Outputs:\n");
    if (config.top_nodes > num_nodes) config.top_nodes = num_nodes;
    for (i = 0; i < config.top_nodes; ++i) {
        printf("Node ID [%d] : Page Rank [%f] \n", node_ids[i], pagerank[i]);
    }

    // clean up
    if (node_ids) { free(node_ids); }
    if (pagerank) { free(pagerank); }
    if (graph_i)  { free(graph_i);  }
    if (graph_o)  { free(graph_o);  }

    return 0;
}
