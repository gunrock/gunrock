/**
 * @brief BC test for shared library
 * @file test_bc.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    // define data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_FLOAT;

    // bc configurations (optional)
    struct GRSetup config;
    config.device     =    0;
    config.src_node   =   -1;  // source vertex to begin search
    config.queue_size = 1.0f;
    config.src_mode   = manually;

    // define graph (undirected graph)
    size_t num_nodes    = 7;
    size_t num_edges    = 26;
    int row_offsets[8]  = {0, 3, 6, 11, 15, 19, 23, 26};
    int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                           5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

    // build graph as input
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    // malloc output graph
    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    // run bc calculations
    gunrock_bc(graph_o, graph_i, config, data_t);

    // test print
    int i;
    printf("Demo Outputs:\n");
    // print per node betweeness centrality values
    float *bc_vals = (float*)malloc(sizeof(float) * graph_i->num_nodes);
    bc_vals = (float*)graph_o->node_values;
    for (i = 0; i < graph_i->num_nodes; ++i) {
        printf("Node_ID [%d] : BC[%f]\n", i, bc_vals[i]);
    }
    printf("\n");
    // print per edge betweeness centrality values
    float *ebc_vals = (float*)malloc(sizeof(float) * graph_i->num_edges);
    ebc_vals = (float*)graph_o->edge_values;
    for (i = 0; i < graph_i->num_edges; ++i) {
        printf("Edge_ID [%d] : EBC[%f]\n", i, ebc_vals[i]);
    }

    // clean up
    if (graph_i) { free(graph_i); }
    if (graph_o) { free(graph_o); }

    return 0;
}
