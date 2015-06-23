/**
 * @brief SSSP test for shared library
 * @file test_sssp.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    // define data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_UINT;

    // configurations (optional)
    struct GRSetup config;
    config.device       =    0;
    config.mark_pred    = true;
    config.queue_size   = 1.0f;
    config.delta_factor =    1;
    config.src_mode     = randomize;

    // define graph
    size_t num_nodes = 7;
    size_t num_edges = 15;

    int row_offsets[8]           = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[15]          = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
    unsigned int edge_values[15] = {39, 6, 41, 51, 63, 17, 10, 44, 41, 13, 58, 43, 50, 59, 35};

    // build graph as input
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];
    graph_i->edge_values = (void*)&edge_values[0];

    // malloc output graph
    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    int *predecessor = (int*)malloc(sizeof(int) * num_nodes);

    // run calculations
    gunrock_sssp(graph_o, predecessor, graph_i, config, data_t);

    // demo test print
    printf("Demo Outputs:\n");
    int *label = (int*)malloc(sizeof(int) * num_nodes);
    label = (int*)graph_o->node_values;
    int node;
    for (node = 0; node < num_nodes; ++node) {
        printf("Node ID [%d] : Label [%d] : Predecessor [%d]\n",
               node, label[node], predecessor[node]);
    }

    // clean up
    if (predecessor) { free(predecessor); }
    if (graph_i) { free(graph_i); }
    if (graph_o) { free(graph_o); }

    return 0;
}
