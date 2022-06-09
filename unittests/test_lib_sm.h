/*
 * @brief Subgraph matching test for shared library advanced interface
 * @file test_lib_sm.h
 */

TEST(sharedlibrary, sm) {
  int num_data_nodes = 5, num_data_edges = 10;
  int data_row_offsets[6] = {0, 2, 6, 7, 9, 10};
  int data_col_indices[10] = {1, 3, 0, 2, 3, 4, 1, 0, 1, 1};

  int num_query_nodes = 3, num_query_edges = 6;
  int query_row_offsets[4] = {0, 2, 4, 6};
  int query_col_indices[6] = {1, 2, 0, 2, 0, 1};

  unsigned long *sm_counts = new unsigned long[1];
  unsigned long **list_sm = new unsigned long*[1];
  unsigned int memspace = 0x01;  // CPU

  double elapsed = sm(num_data_nodes, num_data_edges, data_row_offsets,
                      data_col_indices, num_query_nodes, num_query_edges,
                      query_row_offsets, query_col_indices, 1, sm_counts,
                      list_sm, memspace);


  EXPECT_EQ(sm_counts[0], 1)
      << "Number of matched subgraphs are different";

  unsigned long expected[] = {0, 1, 3};
  for (int i=0; i < 3; i++) {
    EXPECT_EQ((*list_sm)[i], expected[i]);
  }

  delete[] sm_counts;
  sm_counts = NULL;

  delete[] (*list_sm);
  delete[] list_sm;
  list_sm = NULL;
}
