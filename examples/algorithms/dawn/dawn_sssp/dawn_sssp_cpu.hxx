#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thrust/functional.h>
#include <gunrock/formats/formats.hxx>
#include <gunrock/algorithms/algorithms.hxx>
#include "omp.h"

namespace gunrock {
namespace dawn_cpu {
template <typename vertex_t>
void infoprint(vertex_t entry,
               vertex_t total,
               int interval,
               int thread,
               float elapsed_time) {
  if (entry % (total / interval) == 0) {
    float completion_percentage =
        static_cast<float>(entry * 100.0f) / static_cast<float>(total);
    std::cout << "Progress: " << completion_percentage << "%" << std::endl;
    std::cout << "Elapsed Time :"
              << static_cast<double>(elapsed_time) / (thread * 1000) << " s"
              << std::endl;
  }
}

namespace govm_cpu {
using namespace std;
using namespace std::chrono;

template <typename csr_t,
          typename vertex_t,
          typename row_offsets_t,
          typename column_indices_t,
          typename nonzero_values_t,
          typename distances_t,
          typename source_t>
float sssp(csr_t& csr,
           vertex_t& source,
           row_offsets_t& row_offsets,
           column_indices_t& column_indices,
           nonzero_values_t& nonzero_values,
           distances_t& distances,
           source_t& single_source) {
  using weight_t = float;

  int n_vertices = csr.number_of_rows;
  int step = 1;
  bool ptr = false;
  float INF = 1.0 * 0xfffffff;
  float elapsed = 0.0f;
  thrust::host_vector<bool> alpha(n_vertices, 0);
  thrust::host_vector<bool> beta(n_vertices, 0);
  thrust::host_vector<weight_t> result(n_vertices, INF);

  for (int i = row_offsets[source]; i < row_offsets[source + 1]; ++i) {
    result[column_indices[i]] = nonzero_values[i];
    alpha[column_indices[i]] = true;
  }
  auto start = high_resolution_clock::now();
  while (step < n_vertices) {
    step++;
    ptr = false;
    for (int j = 0; j < n_vertices; j++) {
      if (alpha[j]) {
        int start = row_offsets[j];
        int end = row_offsets[j + 1];
        for (int k = start; k < end; k++) {
          int index = column_indices[k];
          float tmp = result[j] + nonzero_values[k];
          if (result[index] > tmp) {
            result[index] = tmp;
            beta[index] = true;
            if ((!ptr) && (index != source))
              ptr = true;
          }
        }
      }
    }
    if (!ptr) {
      break;
    }
    thrust::copy_n(beta.begin(), n_vertices, alpha.begin());
    thrust::fill_n(beta.begin(), n_vertices, false);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  if (single_source == source)
    for (int i = 0; i < n_vertices; i++) {
      if ((result[i] > 0) && (source != i))
        distances.push_back({source, i, result[i]});
    }

  return elapsed;
}

template <typename csr_t, typename vector_t, typename vertex_t>
float apsprun(csr_t& csr, int interval, vector_t& distances, vertex_t& source) {
  float elapsed_time = 0.0;
  int proEntry = 0;
  int num_procs = omp_get_num_procs();  // 获取 CPU 线程数
  printf("Number of available processors: %d\n", num_procs);

  using edge_t = int;
  using weight_t = float;
  int n_vertices = csr.number_of_rows;

  thrust::host_vector<vertex_t> row_offsets = csr.row_offsets;
  thrust::host_vector<vertex_t> column_indices = csr.column_indices;
  thrust::host_vector<weight_t> nonzero_values = csr.nonzero_values;

#pragma omp parallel for
  for (int i = 0; i < n_vertices; i++) {
    if (row_offsets[i] == row_offsets[i + 1]) {
      ++proEntry;
      infoprint(proEntry, n_vertices, interval, num_procs, elapsed_time);
      continue;
    }
    float time = sssp(csr, i, row_offsets, column_indices, nonzero_values,
                      distances, source);
#pragma omp critical
    {
      elapsed_time += time;
      ++proEntry;
    }
    infoprint(proEntry, n_vertices, interval, num_procs, elapsed_time);
  }
  // Output elapsed time and free remaining resources
  elapsed_time = elapsed_time / (num_procs * 1000);
  return (float)elapsed_time;
}

template <typename csr_t, typename vertex_t, typename vector_t>
float sssprun(csr_t& csr, vertex_t& source, vector_t& distances) {
  using weight_t = float;
  thrust::host_vector<vertex_t> row_offsets = csr.row_offsets;
  thrust::host_vector<vertex_t> column_indices = csr.column_indices;
  thrust::host_vector<weight_t> nonzero_values = csr.nonzero_values;
  int n_vertices = csr.number_of_rows;

  int step = 1;
  bool ptr = false;
  thrust::host_vector<bool> alpha(n_vertices, 0);
  thrust::host_vector<bool> beta(n_vertices, 0);

#pragma omp parallel for
  for (int i = row_offsets[source]; i < row_offsets[source + 1]; ++i) {
    distances[column_indices[i]] = nonzero_values[i];
    alpha[column_indices[i]] = true;
  }
  auto t_start = high_resolution_clock::now();
  while (step < n_vertices) {
    step++;
    ptr = false;
#pragma omp parallel for
    for (int j = 0; j < n_vertices; j++) {
      if (alpha[j]) {
        int start = row_offsets[j];
        int end = row_offsets[j + 1];
        for (int k = start; k < end; k++) {
          int index = column_indices[k];
          float tmp = distances[j] + nonzero_values[k];
          if (distances[index] > tmp) {
            distances[index] = thrust::min(distances[index], tmp);
            beta[index] = true;
            if ((!ptr) && (index != source))
              ptr = true;
          }
        }
      }
    }
    if (!ptr) {
      break;
    }
    thrust::copy_n(beta.begin(), n_vertices, alpha.begin());
    thrust::fill_n(beta.begin(), n_vertices, false);
  }
  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}
}  // namespace govm_cpu
}  // namespace dawn_cpu
}  // namespace gun