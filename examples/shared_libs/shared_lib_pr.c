/**
 * @brief PageRank test for shared library advanced interface
 * @file shared_lib_pr.c
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

int main(int argc, char *argv[]) {
  ////////////////////////////////////////////////////////////////////////////

  int num_nodes = 7, num_edges = 26;
  int row_offsets[8] = {0, 3, 6, 11, 15, 19, 23, 26};
  int col_indices[26] = {1, 2, 3, 0, 2, 4, 0, 1, 3, 4, 5, 0, 2,
                         5, 6, 1, 2, 5, 6, 2, 3, 4, 6, 3, 4, 5};

  int *node_ids = (int *)malloc(sizeof(int) * num_nodes);
  float *ranks = (float *)malloc(sizeof(float) * num_nodes);

  double elapsed = pagerank(num_nodes, num_edges, row_offsets, col_indices, 1,
                            node_ids, ranks);

  int node;
  for (node = 0; node < num_nodes; ++node)
    printf("Node_ID [%d] : Score: [%f]\n", node_ids[node], ranks[node]);

  if (node_ids) free(node_ids);
  if (ranks) free(ranks);

  return 0;
}
