/**
 * @brief BC test for shared library advanced interface
 * @file shared_lib_bc.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[]) {
    ////////////////////////////////////////////////////////////////////////////
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_FLOAT;       // attributes type

    struct GRSetup config;                 // gunrock configurations
    int list[] = {0, 1, 2, 3};             // device to run algorithm
    config.num_devices = sizeof(list) / sizeof(list[0]);  // number of devices
    config.device_list = list;             // device list to run algorithm
    config.source_mode = manually;         // manually setting source vertex
    config.source_vertex = 0;              // source vertex to start
    config.max_queue_sizing = 1.0f;        // maximum queue sizing factor

    size_t num_nodes = 7;
    size_t num_edges = 26;
    int row_offsets[8]  = {0, 3, 6, 11, 15, 19, 23, 26};
    int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                           5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    gunrock_bc(graph_o, graph_i, config, data_t);

    ////////////////////////////////////////////////////////////////////////////
    float *scores = (float*)malloc(sizeof(float) * graph_i->num_nodes);
    scores = (float*)graph_o->node_value1;
    size_t node; for (node = 0; node < graph_i->num_nodes; ++node)
        printf("Node_ID [%d] : Score [%.4f]\n", node, scores[node]);

    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);
    if (scores)  free(scores);

    return 0;
}
