#pragma once

#include <chrono>
#include <vector>
#include <cstring>

namespace dawn_bfs_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename vertex_t, typename edge_t>
float run(csr_t& csr,
          vertex_t& single_source,
          vertex_t* distances,
          vertex_t* predecessors) {
  thrust::host_vector<edge_t> _row_offsets(
      csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> _column_indices(csr.column_indices);
  auto row = csr.number_of_rows;
  std::vector<bool> alpha(row, false);
  std::vector<bool> beta(row, false);

  edge_t* row_offsets = _row_offsets.data();
  vertex_t* column_indices = _column_indices.data();

  std::fill_n(distances, row, 0);
  vertex_t step = 1;
  bool is_converged = true;

  for (vertex_t i = row_offsets[single_source];
       i < row_offsets[single_source + 1]; i++) {
    alpha[column_indices[i]] = true;
    distances[column_indices[i]] = 1;
  }

  auto t_start = high_resolution_clock::now();

  while (step < row) {
    step++;
    is_converged = true;
    for (int j = 0; j < row; j++) {
      if (alpha[j]) {
        int start = row_offsets[j];
        int end = row_offsets[j + 1];
        for (int k = start; k < end; k++) {
          if (!distances[column_indices[k]]) {
            distances[column_indices[k]] = step;
            beta[column_indices[k]] = 1;
            is_converged = false;
          }
        }
      }
    }
    if (is_converged) {
      break;
    }
    std::copy(beta.begin(), beta.end(), alpha.begin());
    std::fill_n(beta.begin(), row, false);
  }

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();

  // Here we reset the values in the distances vector to accommodate the check
  // function. This is because the initialization method of the distances vector
  // in dawn_bfs_cpu differs from that of other functions.
  for (vertex_t i = 0; i < row; i++)
    if (distances[i] == 0)
      distances[i] = std::numeric_limits<vertex_t>::max();
  distances[single_source] = 0;

  return (float)elapsed / 1000;
}

}  // namespace dawn_bfs_cpu

namespace dawn_sssp_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float run(csr_t& csr,
          vertex_t& single_source,
          weight_t* distances,
          vertex_t* predecessors) {
  thrust::host_vector<edge_t> _row_offsets(
      csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> _column_indices(csr.column_indices);
  thrust::host_vector<weight_t> _nonzero_values(csr.nonzero_values);

  edge_t* row_offsets = _row_offsets.data();
  vertex_t* column_indices = _column_indices.data();
  weight_t* nonzero_values = _nonzero_values.data();

  auto row = csr.number_of_rows;
  std::vector<bool> alpha(row, false);
  std::vector<bool> beta(row, false);
  std::fill_n(distances, row, std::numeric_limits<weight_t>::max());
  vertex_t step = 1;
  bool is_converged = true;
  for (vertex_t i = row_offsets[single_source];
       i < row_offsets[single_source + 1]; i++) {
    alpha[column_indices[i]] = true;
    distances[column_indices[i]] = nonzero_values[i];
  }
  distances[single_source] = 0;

  auto t_start = high_resolution_clock::now();
  while (step < row) {
    step++;
    is_converged = true;
    for (int j = 0; j < row; j++) {
      if (alpha[j]) {
        int start = row_offsets[j];
        int end = row_offsets[j + 1];
        if (start != end) {
          for (int k = start; k < end; k++) {
            int index = column_indices[k];
            float tmp = distances[j] + nonzero_values[k];
            if ((distances[index] > tmp) && (index != single_source)) {
              distances[index] = std::min(distances[index], tmp);
              beta[index] = true;
              if (is_converged)
                is_converged = false;
            }
          }
        }
      }
    }

    if (is_converged) {
      break;
    }
    std::copy(beta.begin(), beta.end(), alpha.begin());
    std::fill_n(beta.begin(), row, false);
  }
  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();

  return (float)elapsed / 1000;
}

}  // namespace dawn_sssp_cpu
