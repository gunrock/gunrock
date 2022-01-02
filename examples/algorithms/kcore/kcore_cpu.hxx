#pragma once

#include <chrono>

namespace kcore_cpu {

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float run(csr_t& csr, int* k_cores) {
  thrust::host_vector<edge_t> row_offsets(csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> column_indices(csr.column_indices);
  thrust::host_vector<weight_t> nonzero_values(csr.nonzero_values);

  // Initialize data
  std::vector<int> remaining;
  std::vector<int> remaining_buff;
  std::vector<int> to_be_deleted;
  thrust::fill(k_cores, k_cores + csr.number_of_rows, 0);
  thrust::host_vector<int> degrees(csr.number_of_rows);
  for (int v = 0; v < csr.number_of_rows; v++) {
    degrees[v] = row_offsets[v + 1] - row_offsets[v];
    if (degrees[v] != 0) {
      remaining.push_back(v);
    }
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  for (int k = 1; k < csr.number_of_rows; k++) {
    while (true) {
      // delete vertices with degree <= k
      for (auto v : remaining) {
        if (degrees[v] <= k) {
          k_cores[v] = k;
          to_be_deleted.push_back(v);
        } else {
          remaining_buff.push_back(v);
        }
      }
      remaining.swap(remaining_buff);
      remaining_buff.clear();
      if (to_be_deleted.empty())
        break;  // increment k when all vertices have degree > k
      // decrement degree of deleted vertices' neighbors
      for (auto v : to_be_deleted) {
        vertex_t start = row_offsets[v];
        vertex_t end = row_offsets[v + 1];
        for (vertex_t i = start; i < end; i++) {
          degrees[column_indices[i]]--;
        }
      }
      to_be_deleted.clear();
    }
    if (remaining.empty())
      break;  // stop when graph is empty
  }

  auto t_stop = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start)
          .count();
  return (float)elapsed / 1000;
}

}  // namespace kcore_cpu
