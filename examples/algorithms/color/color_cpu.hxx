#pragma once

#include <chrono>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <gunrock/algorithms/generate/random.hxx>

namespace color_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float run(csr_t& csr, vertex_t* colors) {
  thrust::host_vector<edge_t> row_offsets(csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> column_indices(csr.column_indices);

  auto t_start = high_resolution_clock::now();

  vertex_t n_vertices = csr.number_of_rows;
  for (vertex_t i = 0; i < n_vertices; i++)
    colors[i] = -1;

  thrust::host_vector<weight_t> randoms(n_vertices);
  gunrock::generate::random::uniform_distribution(randoms);

  int color = 0;
  int n_left = n_vertices;
  while (n_left > 0) {
    for (vertex_t v = 0; v < n_vertices; v++) {
      if (colors[v] != -1)
        continue;

      edge_t start_edge = row_offsets[v];
      edge_t num_neighbors = row_offsets[v + 1] - row_offsets[v];

      bool colormax = true;
      bool colormin = true;

      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        vertex_t u = column_indices[e];

        if ((colors[u] != -1) && (colors[u] != color) &&
                (colors[u] != color + 1) ||
            (v == u))
          continue;

        if (randoms[v] <= randoms[u])
          colormax = false;
        if (randoms[v] >= randoms[u])
          colormin = false;
      }

      if (colormax) {
        colors[v] = color;
      } else if (colormin) {
        colors[v] = color + 1;
      }

      if (colormax || colormin)
        n_left--;
    }

    color += 2;
  }

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float compute_error(csr_t& csr,
                    thrust::device_vector<vertex_t> _gpu_colors,
                    thrust::host_vector<vertex_t> cpu_colors) {
  thrust::host_vector<edge_t> row_offsets(csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> column_indices(csr.column_indices);
  thrust::host_vector<vertex_t> gpu_colors(_gpu_colors);

  vertex_t n_vertices = csr.number_of_rows;

  vertex_t cpu_errors = 0;
  vertex_t gpu_errors = 0;

  for (vertex_t v = 0; v < n_vertices; v++) {
    for (edge_t e = row_offsets[v]; e < row_offsets[v + 1]; e++) {
      vertex_t u = column_indices[e];

      // Do not check self-loops.
      if (v == u)
        continue;

      // Check if colors are the same among neighborhoods.
      if (gpu_colors[u] == gpu_colors[v] || gpu_colors[v] == -1) {
        std::cout << "Error: " << v << " " << u << " " << gpu_colors[v] << " "
                  << gpu_colors[u] << std::endl;
        gpu_errors++;
      }
      // if(cpu_colors[u] == cpu_colors[v] || cpu_colors[v] == -1) cpu_errors++;
    }
  }

  return gpu_errors + cpu_errors;
}

}  // namespace color_cpu