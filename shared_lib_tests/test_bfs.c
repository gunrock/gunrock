/**
 * @brief BFS test for shared library
 * @file test_bfs.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    // define data types
    struct GRTypes data_t;
    data_t.VTXID_TYPE = VTXID_INT;
    data_t.SIZET_TYPE = SIZET_INT;
    data_t.VALUE_TYPE = VALUE_INT;

    // bfs configurations (optional)
    struct GRSetup config;
    config.device      = 0;
    config.src_mode    = randomize;
    config.src_node    = 1;      // source vertex to begin search
    config.mark_pred   = false;  // do not mark predecessors
    config.idempotence = false;  // wether enable idempotence
    config.queue_size  = 1.0f;

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

    // run bfs calculations
    gunrock_bfs(graph_o, graph_i, config, data_t);

    // test print
    int i;
    printf("Demo Outputs:\n");
    int *labels = (int*)malloc(sizeof(int) * graph_i->num_nodes);
    labels = (int*)graph_o->node_values;
    for (i = 0; i < graph_i->num_nodes; ++i) {
        printf("Node_ID [%d] : Label [%d]\n", i, labels[i]);
    }

    // clean up
    if (graph_i) { free(graph_i); }
    if (graph_o) { free(graph_o); }
    if (labels)  { free(labels);  }

    return 0;
}
