/**
 * @brief MST test for shared library
 * @file test_mst.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    // set problem data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.VALUE_TYPE = VALUE_INT;
    data_t.SIZET_TYPE = SIZET_INT;

    // configurations (optional)
    struct GRSetup config;
    config.device = 0;

    // tiny sample graph
    size_t num_nodes = 7;
    size_t num_edges = 26;
    int row_offsets[8]  = {0, 3, 6, 11, 15, 19, 23, 26};
    int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                           5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};
    int edge_values[26] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    // build an graph as input
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];
    graph_i->edge_values = (void*)&edge_values[0];

    // malloc output graph
    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    // call minimum spanning tree
    gunrock_mst(graph_o, graph_i, config, data_t);

    // demo test print
    printf("Demo Outputs:\n");
    int *mst_mask = (int*)malloc(sizeof(int) * num_edges);
    mst_mask = (int*)graph_o->edge_values;
    int edge;
    for (edge = 0; edge < num_edges; ++edge) {
        printf("Edge ID [%d] : Mask [%d]\n", edge, mst_mask[edge]);
    }

    // clean up
    if (graph_i) { free(graph_i); }
    if (graph_o) { free(graph_o); }

    return 0;
}
