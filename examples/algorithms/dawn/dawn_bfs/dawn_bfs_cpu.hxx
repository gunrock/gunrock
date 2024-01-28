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

namespace sovm_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t,
          typename vertex_t,
          typename row_offsets_t,
          typename column_indices_t,
          typename distances_t,
          typename index_t>
float sssp(csr_t& csr,
           vertex_t& source,
           row_offsets_t& row_offsets,
           column_indices_t& column_indices,
           distances_t& distances,
           index_t& index) {
  int step = 1;
  float elapsed = 0.0f;
  int alphaPtr = row_offsets[source + 1] - row_offsets[source];
  int Ptr = 0;
  vertex_t n_vertices = csr.number_of_rows;
  thrust::host_vector<vertex_t> alpha(n_vertices, 0);
  thrust::host_vector<vertex_t> result(n_vertices, 0);
  thrust::host_vector<vertex_t> beta(n_vertices, 0);

  for (int i = row_offsets[source]; i < row_offsets[source + 1]; i++) {
    result[column_indices[i]] = 1;
    alpha[i - row_offsets[source]] = column_indices[i];
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < n_vertices) {
    step++;
    Ptr = 0;
    for (int j = 0; j < alphaPtr; j++) {
      int start = row_offsets[alpha[j]];
      int end = row_offsets[alpha[j] + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!result[column_indices[k]]) {
            beta[Ptr] = column_indices[k];
            Ptr++;
            result[column_indices[k]] = step;
          }
        }
      }
    }
    thrust::copy_n(beta.begin(), Ptr, alpha.begin());
    alphaPtr = Ptr;
    if (!alphaPtr) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_tmp = end - start;
  elapsed += elapsed_tmp.count();

  if (index == source)
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

#pragma omp parallel for
  for (int i = 0; i < n_vertices; i++) {
    if (row_offsets[i] == row_offsets[i + 1]) {
      ++proEntry;
      infoprint(proEntry, n_vertices, interval, num_procs, elapsed_time);
      continue;
    }
    float time = sssp(csr, i, row_offsets, column_indices, distances, source);
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
  using weight_t = typename csr_t::value_type;

  thrust::host_vector<vertex_t> row_offsets = csr.row_offsets;
  thrust::host_vector<vertex_t> column_indices = csr.column_indices;

  int step = 1;
  bool ptr = false;
  int n_vertex = csr.number_of_rows;
  thrust::host_vector<vertex_t> alpha(n_vertex, 0);
  thrust::host_vector<vertex_t> beta(n_vertex, 0);

#pragma omp parallel for
  for (int i = row_offsets[source]; i < row_offsets[source + 1]; ++i) {
    distances[column_indices[i]] = 1;
    alpha[column_indices[i]] = true;
    beta[column_indices[i]] = true;
  }
  auto t_start = high_resolution_clock::now();
  while (step < n_vertex) {
    step++;
    ptr = false;
#pragma omp parallel for
    for (int j = 0; j < n_vertex; j++) {
      if (alpha[j]) {
        int start = row_offsets[j];
        int end = row_offsets[j + 1];
        if (start != end) {
          for (int k = start; k < end; k++) {
            if (!distances[column_indices[k]]) {
              distances[column_indices[k]] = step;
              if (!ptr)
                ptr = true;
            }
          }
        }
      }
    }
#pragma omp parallel for
    for (int j = 0; j < n_vertex; j++) {
      if (distances[j] && (!beta[j])) {
        alpha[j] = true;
        beta[j] = true;
      } else {
        alpha[j] = false;
      }
    }
    if (!ptr) {
      break;
    }
  }
  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace sovm_cpu

namespace bovm_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename input_t, typename distances_t>
float sssprun(csr_t& csr, input_t& input, distances_t& distances) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  thrust::host_vector<vertex_t> row_offsets = csr.row_offsets;
  thrust::host_vector<vertex_t> column_indices = csr.column_indices;

  int step = 1;
  bool ptr = false;
  int n_vertices = csr.number_of_rows;
  thrust::host_vector<bool> alpha(n_vertices, 0);
  thrust::host_vector<bool> beta(n_vertices, 0);
  thrust::copy(input.begin(), input.end(), alpha.begin());
  thrust::copy(input.begin(), input.end(), beta.begin());

  auto t_start = high_resolution_clock::now();
  while (step < n_vertices) {
    step++;
    ptr = false;
#pragma omp parallel for
    for (int j = 0; j < n_vertices; j++) {
      if (!alpha[j]) {
        int start = row_offsets[j];
        int end = row_offsets[j + 1];
        if (start != end) {
          for (int k = start; k < end; k++) {
            if (alpha[column_indices[k]]) {
              beta[j] = true;
              distances[j] = step;
              if (!ptr)
                ptr = true;
              break;
            }
          }
        }
      }
    }
    thrust::copy(beta.begin(), beta.end(), alpha.begin());
    if (!ptr)
      break;
  }
  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}
}  // namespace bovm_cpu
}  // namespace dawn_cpu
}  // namespace gunrock