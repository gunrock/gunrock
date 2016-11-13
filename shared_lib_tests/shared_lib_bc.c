/**
 * @brief BC test for shared library advanced interface
 * @file shared_lib_bc.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char* argv[])
{
    ////////////////////////////////////////////////////////////////////////////
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_FLOAT;       // attributes type

    struct GRSetup *config = InitSetup(1, NULL);   // gunrock configurations

    int num_nodes = 7, num_edges = 26;
    int row_offsets[8]  = {0, 3, 6, 11, 15, 19, 23, 26};
    int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                           5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graphi->num_nodes   = num_nodes;
    graphi->num_edges   = num_edges;
    graphi->row_offsets = (void*)&row_offsets[0];
    graphi->col_indices = (void*)&col_indices[0];

    gunrock_bc(grapho, graphi, config, data_t);

    ////////////////////////////////////////////////////////////////////////////
    float *scores = (float*)malloc(sizeof(float) * graphi->num_nodes);
    scores = (float*)grapho->node_value1;
    int node; for (node = 0; node < graphi->num_nodes; ++node)
        printf("Node_ID [%d] : Score [%.4f]\n", node, scores[node]);

    if (graphi) free(graphi);
    if (grapho) free(grapho);
    if (scores) free(scores);

    return 0;
}
