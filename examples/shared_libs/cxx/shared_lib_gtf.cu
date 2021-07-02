/**
 * @brief SSSP test for shared library CXX interface
 * @file shared_lib_hits.cu
 */

#include <stdio.h>

#include <cmath>

#include <gunrock/gunrock.hpp>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>


int main(int argc, char** argv) {
  int num_nodes = 7, num_edges = 15;
  int row_offsets[8] = {0, 3, 6, 9, 11, 14, 15, 15};
  int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
  const float edge_values[15] = {39, 6,  41, 51, 63, 17, 10, 44,
                                 41, 13, 58, 43, 50, 59, 35};
  int source = 1;
  int sink = 5;
  const bool mark_pred = false;

  unsigned int memspace = gunrock::util::HOST;
  float *flow = (float *)malloc(num_nodes * sizeof(float));
  float *residuals = (float *)NULL;

  double elapsed =
      gtf<int, int, float>(num_nodes, num_edges, row_offsets, col_indices, edge_values,
           1 /* num runs */, source, sink, flow, residuals, memspace);

  printf("residuals: ");
  for (int i = 0; i < num_nodes; i++) {
    printf("%f ", residuals[i]);
  }
  printf("\n\n");

  memspace = gunrock::util::DEVICE;
  thrust::device_vector<int> device_row_offsets(row_offsets, row_offsets + 8);
  thrust::device_vector<int> device_col_indices(col_indices,
                                                col_indices + 15);
  thrust::device_vector<float> device_edge_values(edge_values,
                                                  edge_values + 15);
  thrust::device_vector<float> device_flow(num_nodes);
  thrust::device_vector<float> device_residuals(1);

  elapsed =
      gtf<int, int, float>(num_nodes, num_edges, device_row_offsets.data().get(),
           device_col_indices.data().get(), device_edge_values.data().get(),
           1, /* num runs */ source, sink, device_flow.data().get(),
           device_residuals.data().get(), memspace);

  printf("residuals: ");
  thrust::copy(device_residuals.begin(), device_residuals.end(),
               std::ostream_iterator<float>(std::cout, " "));

  printf("\n");
  return 0;

}
