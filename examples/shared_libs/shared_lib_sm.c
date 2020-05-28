/*
 * @brief Subgraph matching test for shared library advanced interface
 * @file shared_lib_sm.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char *argv[]) {
  int num_data_nodes = 5, num_data_edges = 10;
  int data_row_offsets[6] = {0, 2, 6, 7, 9, 10};
  int data_col_indices[10] = {1, 3, 0, 2, 3, 4, 1, 0, 1, 1};

  int num_query_nodes = 3, num_query_edges = 6;
  int query_row_offsets[4] = {0, 2, 4, 6};
  int query_col_indices[6] = {1, 2, 0, 2, 0, 1};
  unsigned int device = 0x01;  // CPU

  unsigned long *sm_counts = (unsigned long *)malloc(sizeof(unsigned long));
  unsigned long *list_sm;

  double elapsed =
      sm(num_data_nodes, num_data_edges, data_row_offsets, data_col_indices,
         num_query_nodes, num_query_edges, query_row_offsets, query_col_indices,
         1, sm_counts, list_sm, device);

  printf("Number matched subgraphs: [%d]\n", sm_counts[0]);

  if (sm_counts) free(sm_counts);
  if (list_sm) free(list_sm);

  return 0;
}
