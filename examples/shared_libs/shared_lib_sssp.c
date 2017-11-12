/**
 * @brief BFS test for shared library advanced interface
 * @file shared_lib_bfs.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
    ////////////////////////////////////////////////////////////////////////////
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_INT;         // attributes type
    int srcs[1] = {0};

    struct GRSetup *config = InitSetup(1, srcs);   // gunrock configurations

    int num_nodes = 7, num_edges = 15;  // number of nodes and edges
    int row_offsets[8]  = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
    int edge_values[15] = {39, 6, 41, 51, 63, 17, 10, 44, 41, 13, 58, 43, 50, 59, 35};

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graphi->num_nodes   = num_nodes;
    graphi->num_edges   = num_edges;
    graphi->row_offsets = (void*)&row_offsets[0];
    graphi->col_indices = (void*)&col_indices[0];
    graphi->edge_values = (void*)&edge_values[0];

    gunrock_sssp(grapho, graphi, config, data_t);

    ////////////////////////////////////////////////////////////////////////////
    int *labels = (int*)malloc(sizeof(int) * graphi->num_nodes);
    labels = (int*)grapho->node_value1;
    int node; for (node = 0; node < graphi->num_nodes; ++node)
        printf("Node_ID [%d] : Label [%d]\n", node, labels[node]);

    if (graphi) free(graphi);
    if (grapho) free(grapho);
    if (labels) free(labels);

    return 0;
}
