/*
 * @brief Subgraph matching test for shared library advanced interface
 * @file test_ssspapi.cu
 */

#include <stdio.h>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gunrock/app/sssp/sssp_app.cu>

int main(int argc, char *argv[]) {
  int num_nodes = 7, num_edges = 15;
  int row_offsets[8] = {0, 3, 6, 9, 11, 14, 15, 15};
  int col_indices[15] = {1, 2, 3, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
  const float edge_values[15] = {39, 6,  41, 51, 63, 17, 10, 44,
                                 41, 13, 58, 43, 50, 59, 35};
  int source = 1;
  const bool mark_pred = false;

  unsigned int memspace;

  if (*argv[1] == '1') {
    memspace = 0x01;
    float *distances = (float *)malloc(num_nodes * sizeof(float));
    int *preds = (int *)NULL;

    double elapsed =
        sssp(num_nodes, num_edges, row_offsets, col_indices, edge_values,
             source, mark_pred, distances, preds, memspace);

    printf("Distances: ");
    for (int i = 0; i < num_nodes; i++) {
      printf("%f ", distances[i]);
    }
    printf("\n");
  }

  if (*argv[1] == '2') {
    memspace = 0x02;
    thrust::device_vector<int> device_row_offsets(row_offsets, row_offsets + 8);
    thrust::device_vector<int> device_col_indices(col_indices,
                                                  col_indices + 15);
    thrust::device_vector<float> device_edge_values(edge_values,
                                                    edge_values + 15);
    thrust::device_vector<float> device_distances(num_nodes);
    thrust::device_vector<int> device_preds(1);

    double elapsed =
        sssp(num_nodes, num_edges, device_row_offsets.data().get(),
             device_col_indices.data().get(), device_edge_values.data().get(),
             source, mark_pred, device_distances.data().get(),
             device_preds.data().get(), memspace);

    printf("Distances: ");
    thrust::copy(device_distances.begin(), device_distances.end(),
                 std::ostream_iterator<int>(std::cout, " "));
  }

  return 0;
}
