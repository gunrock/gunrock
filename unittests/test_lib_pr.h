/*
 * @brief PageRank test for shared library advanced interface
 * @file test_lib_pr.h
 */

TEST(sharedlibrary, pagerank) {

  int num_nodes = 7, num_edges = 26;
  int row_offsets[8]  = {0, 3, 6, 11, 15, 19, 23, 26};
  int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                         5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

  int 	*top_nodes = new int  [num_nodes];
  float	*top_ranks = new float[num_nodes];

  double elapsed =  pagerank(num_nodes, num_edges, row_offsets, 
		  	     col_indices, 1, top_nodes, top_ranks); 

  double nodes[7] = {2, 3, 4, 5, 0, 1, 6};
  double scores[7] = {0.186179, 0.152261, 0.152261, 0.151711,
    0.119455, 0.119455, 0.118680};

  for (int node = 0; node < num_nodes; ++node) {
    EXPECT_EQ(top_nodes[node], nodes[node])
      << "Node indices differ at node index " << node;
    EXPECT_NEAR(top_ranks[node], scores[node], 0.0000005)
      << "Scores differ at node index " << node;
  }

  delete[] top_nodes; top_nodes = NULL;
  delete[] top_ranks; top_ranks = NULL;

}
