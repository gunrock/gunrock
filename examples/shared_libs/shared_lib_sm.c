/*
 * @brief Subgraph matching test for shared library advanced interface
 * @file shared_lib_sm.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char *argv[]) {

  int num_data_nodes = 5, num_data_edges = 5;
  int data_row_offsets[6]  = {0, 2, 6, 7, 9};
  int data_col_indices[5] = {1, 3, 0, 2, 3};

  int num_query_nodes = 3, num_query_edges = 3;
  int query_row_offsets[4]  = {0, 2, 4, 6};
  int query_col_indices[3] = {1, 2, 0};

  int *sm_counts = (int *)malloc(sizeof(int) * num_data_nodes);

  double elapsed =  sm(num_data_nodes, num_data_edges, data_row_offsets, 
		       data_col_indices, num_query_nodes, num_query_edges,
                       query_row_offsets, query_col_indices, 1, sm_counts); 

  int node;
  for (node = 0; node < num_data_nodes; ++node) {
      printf("Node_ID [%d] : Number matched subgraphs: [%d]\n", node, sm_counts[node]);
  }

  if (sm_counts) free(sm_counts);

  return 0;
}
