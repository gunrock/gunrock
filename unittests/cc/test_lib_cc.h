/**
 * @brief CC test for shared library advanced interface
 * @file test_lib_cc.h
 */

namespace gunrock {

TEST(sharedlibrary, connectedcomponent)
{
    struct GRTypes data_t;                 // data type structure
    data_t.VTXID_TYPE = VTXID_INT;         // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;         // graph size type
    data_t.VALUE_TYPE = VALUE_INT;         // attributes type
    int srcs[4] = {0,2,4,5};

    struct GRSetup *config = InitSetup(4, srcs);   // gunrock configurations

    // graph is:
    // 0->1->2; 3->4; 5; 6->7<-8; 9<-10->11; 12->13->14->12
    const int row_offsets[] =
        {0, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 7, 7, 8, 9, 10};
    const int num_nodes = (sizeof(row_offsets) / sizeof(row_offsets[0])) - 1;
    const int col_indices[] = {1, 2, 4, 7, 7, 9, 11, 13, 14, 12};
    const int num_edges = sizeof(col_indices) / sizeof(col_indices[0]);

    struct GRGraph *grapho = (struct GRGraph*) malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*) malloc(sizeof(struct GRGraph));
    graphi->num_nodes   = num_nodes;
    graphi->num_edges   = num_edges;
    graphi->row_offsets = (void*) &row_offsets[0];
    graphi->col_indices = (void*) &col_indices[0];

    gunrock_cc(grapho, graphi, config, data_t);

    int *labels = (int*) malloc(sizeof(int) * graphi->num_nodes);
    labels = (int*)grapho->node_value1;
    int result[num_nodes] = {0, 0, 0, 3, 3, 5, 6, 6, 6, 9, 9, 9, 12, 12, 12};

    for (int i = 0; i < graphi->num_nodes; ++i) {
        EXPECT_EQ(labels[i], result[i]) << "At index " << i
                                        << ", computedlabel "
                                        << labels[i]
                                        << " differs from known label "
                                        << result[i];
    }

    free(graphi);
    free(grapho);
    free(labels);

}
} // namespace gunrock
