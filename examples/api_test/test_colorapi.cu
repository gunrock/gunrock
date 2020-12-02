/*
 * @brief Subgraph matching test for shared library advanced interface
 * @file test_ssspapi.cu
 */

#include <stdio.h>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gunrock/gunrock.hpp>

int main(int argc, char *argv[]) {
  // Test graph, toy example.
  int num_nodes = 7, num_edges = 15;
  int row_offsets[8] = {0, 3, 6, 9, 11, 14, 15, 15};
  int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};

  int *colors = (int *)malloc(num_nodes * sizeof(int));
  int num_colors = 0;

  // See gunrock.hxx file.
  double elapsed =
      color(num_nodes, num_edges, row_offsets, col_indices, &colors,
            &num_colors, 1 /*num_runs*/);

  // Output, host side.
  printf("Colors: ");
  for (int i = 0; i < num_nodes; i++) {
    printf("%u ", colors[i]);
  }
  printf("\n");
  printf("Number of Colors: %u\n", num_colors);

  return 0;
}
