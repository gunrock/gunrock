/**
 * @brief BFS test for shared library advanced interface
 * @file test_lib_bfs.h
 */

#include <stdio.h>
#include <gunrock/gunrock.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace gunrock {

TEST(sharedlibrary, breadthfirstsearch) 
{
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_INT;         // attributes type
    int srcs[3] = {0,1,2};

    struct GRSetup *config = InitSetup(3, srcs);   // gunrock configurations

    int num_nodes = 7, num_edges = 15;  // number of nodes and edges
    int row_offsets[8]  = {0, 3, 6, 9, 11, 14, 15, 15};
    int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graphi->num_nodes   = num_nodes;
    graphi->num_edges   = num_edges;
    graphi->row_offsets = (void*)&row_offsets[0];
    graphi->col_indices = (void*)&col_indices[0];

    gunrock_bfs(grapho, graphi, config, data_t);

    int *labels = (int*)malloc(sizeof(int) * graphi->num_nodes);
    labels = (int*)grapho->node_value1;
    int result[7] = {2147483647, 2147483647, 0, 1, 1, 1, 2};

    for (int i = 0; i < graphi->num_nodes; ++i) {
      EXPECT_EQ(labels[i], result[i]) << "Vectors x and y differ at index " << i;
    }

    if (graphi) free(graphi);
    if (grapho) free(grapho);
    if (labels) free(labels);

}
} // namespace gunrock
