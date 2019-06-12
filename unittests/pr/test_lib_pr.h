/*
 * @brief PageRank test for shared library advanced interface
 * @file test_lib_pr.h
 */

namespace gunrock {

TEST(sharedlibrary, pagerank) {
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

  gunrock_pagerank(grapho, graphi, config, data_t);

  ////////////////////////////////////////////////////////////////////////////
  // int   *top_nodes = (  int*)malloc(sizeof(  int) * graphi->num_nodes);
  // float *top_ranks = (float*)malloc(sizeof(float) * graphi->num_nodes);
  // top_nodes = (  int*)grapho->node_value2;
  // top_ranks = (float*)grapho->node_value1;
  int   *top_nodes = (  int*)grapho->node_value2;
  float *top_ranks = (float*)grapho->node_value1;

  double nodes[7] = {2, 3, 4, 5, 0, 1, 6};
  double scores[7] = {0.186179, 0.152261, 0.152261, 0.151711,
    0.119455, 0.119455, 0.118680};

  for (int node = 0; node < config -> top_nodes; ++node) {
    // printf("Node_ID [%d] : Score: [%f]\n", node_ids[node], ranks[node]);
    EXPECT_EQ(top_nodes[node], nodes[node])
      << "Node indices differ at node index " << node;
    EXPECT_NEAR(top_ranks[node], scores[node], 0.0000005)
      << "Scores differ at node index " << node;
  }

  if (graphi) free(graphi);
  if (grapho) free(grapho);
  if (top_nodes) free(top_nodes);
  if (top_ranks) free(top_ranks);
}
}  // namespace gunrock
