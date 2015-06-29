/**
 * @brief CC test for shared library
 * @file test_cc.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    // define data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_INT;

    // connected component configurations
    struct GRSetup config;
    config.device = 0;

    // define graph
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
    unsigned int *components = (unsigned int*)malloc(sizeof(unsigned int));

    // run connected component calculations
    gunrock_cc(graph_o, components, graph_i, config, data_t);

    // demo test print
    printf("Number of Components: %d\n", components[0]);
    printf("Demo Outputs:\n");
    int *component_ids = (int*)malloc(sizeof(int) * graph_i->num_nodes);
    component_ids = (int*)graph_o->node_values;
    int node;
    for (node = 0; node < graph_i->num_nodes; ++node) {
        printf("Node_ID [%d] : Component_ID [%d]\n", node, component_ids[node]);
    }

    // clean up
    if (graph_i) { free(graph_i); }
    if (graph_o) { free(graph_o); }

    return 0;
}
