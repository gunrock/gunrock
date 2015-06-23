/**
 * @brief Top K test for shared library
 * @file test_topk.c
 */

#include <gunrock/gunrock.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    // define data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_INT;

    struct GRSetup config;
    config.device    = 0;
    config.top_nodes = 3;

    // define graph (directed, reversed and non-reversed)
    size_t num_nodes = 7;
    size_t num_edges = 15;

    int row_offsets[8]  = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};

    int col_offsets[8]  = {0, 1, 2, 5, 7, 9, 12, 15};
    int row_indices[15] = {1, 0, 0, 1, 4, 0, 2, 1, 2, 2, 3, 4, 3, 4, 5};

    // build graph as input
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes = num_nodes;
    graph_i->num_edges = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];
    graph_i->col_offsets = (void*)&col_offsets[0];
    graph_i->row_indices = (void*)&row_indices[0];

    // malloc output result arrays
    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    int *node_ids    = (int*)malloc(sizeof(int) * config.top_nodes);
    int *in_degrees  = (int*)malloc(sizeof(int) * config.top_nodes);
    int *out_degrees = (int*)malloc(sizeof(int) * config.top_nodes);

    // run topk calculations
    gunrock_topk(
        graph_o, node_ids, in_degrees, out_degrees, graph_i, config, data_t);

    // print results for check correctness
    printf("Demo Outputs:\n");
    int node;
    for (node = 0; node < config.top_nodes; ++node) {
        printf("Node ID [%d] : in_degrees [%d] : out_degrees [%d] \n",
               node_ids[node], in_degrees[node], out_degrees[node]);
    }

    // clean up
    if (in_degrees)  free(in_degrees);
    if (out_degrees) free(out_degrees);
    if (node_ids)    free(node_ids);
    if (graph_i)     free(graph_i);
    if (graph_o)     free(graph_o);

    return 0;
}