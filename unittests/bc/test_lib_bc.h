/**
 * @brief BC test for shared library advanced interface
 * @file test_lib_bc.h
 */

// add to gunrock's namespace
namespace gunrock {

TEST(sharedlibrary, betweennesscentrality) {
  struct GRTypes data_t;            // data type structure
  data_t.VTXID_TYPE = VTXID_INT;    // vertex identifier
  data_t.SIZET_TYPE = SIZET_INT;    // graph size type
  data_t.VALUE_TYPE = VALUE_FLOAT;  // attributes type

  struct GRSetup *config = InitSetup(1, NULL);  // gunrock configurations

  int num_nodes = 7, num_edges = 26;
  int row_offsets[8] = {0, 3, 6, 11, 15, 19, 23, 26};
  int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                         5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

  struct GRGraph *grapho = (struct GRGraph *)malloc(sizeof(struct GRGraph));
  struct GRGraph *graphi = (struct GRGraph *)malloc(sizeof(struct GRGraph));
  graphi->num_nodes = num_nodes;
  graphi->num_edges = num_edges;
  graphi->row_offsets = (void *)&row_offsets[0];
  graphi->col_indices = (void *)&col_indices[0];

  gunrock_bc(grapho, graphi, config, data_t);

  float *scores = (float *)malloc(sizeof(float) * graphi->num_nodes);
  scores = (float *)grapho->node_value1;

  float results[7] = {0.00, 0.25, 0.50, 0.75, 0.00, 0.00, 0.00};

  for (int node = 0; node < graphi->num_nodes; ++node) {
    EXPECT_EQ(scores[node], results[node])
        << "Vectors x and y differ at index " << node;
  }

  if (graphi) free(graphi);
  if (grapho) free(grapho);
  if (scores) free(scores);
}
}  // namespace gunrock
