/**
 * @file benchmark.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2023-03-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <vector>

#include <gunrock/cuda/cuda.hxx>
#include <gunrock/util/math.hxx>

namespace gunrock {
namespace benchmark {

class benchmark_t {
 public:
  benchmark_t()
      : edges_visited(1),
        vertices_visited(1),
        search_depth(0),
        total_runtime() {}

  thrust::device_vector<unsigned int> edges_visited;
  thrust::device_vector<unsigned int> vertices_visited;

  thrust::host_vector<unsigned int> h_edges_visited;
  thrust::host_vector<unsigned int> h_vertices_visited;

  std::size_t search_depth;
  double total_runtime;
};

struct device_benchmark_t {
  unsigned int* d_edges_visited;
  unsigned int* d_vertices_visited;
};

struct host_benchmark_t {
  unsigned int edges_visited = 0;
  unsigned int vertices_visited = 0;
  std::size_t search_depth = 0;
  double total_runtime = 0;
};

benchmark_t benchmark_instance;
__managed__ device_benchmark_t BXXX;

__device__ void LOG_EDGE_VISITED(size_t edges) {
  math::atomic::add(BXXX.d_edges_visited, static_cast<unsigned int>(edges));
}

__device__ void LOG_VERTEX_VISITED(size_t vertices) {
  math::atomic::add(BXXX.d_vertices_visited,
                    static_cast<unsigned int>(vertices));
}

void INIT_BENCH() {
#if ESSENTIALS_COLLECT_METRICS
  thrust::fill(benchmark_instance.edges_visited.begin(), benchmark_instance.edges_visited.end(), 0);
  thrust::fill(benchmark_instance.vertices_visited.begin(), benchmark_instance.vertices_visited.end(), 0);

  BXXX.d_edges_visited = benchmark_instance.edges_visited.data().get();
  BXXX.d_vertices_visited = benchmark_instance.vertices_visited.data().get();
#endif
}

void DESTROY_BENCH() {
#if ESSENTIALS_COLLECT_METRICS
  BXXX.d_edges_visited = nullptr;
  BXXX.d_vertices_visited = nullptr;
#endif
}

host_benchmark_t EXTRACT() {
  host_benchmark_t results;
#if ESSENTIALS_COLLECT_METRICS
  benchmark_instance.h_edges_visited = benchmark_instance.edges_visited;
  benchmark_instance.h_vertices_visited = benchmark_instance.vertices_visited;
  results.edges_visited = benchmark_instance.h_edges_visited[0];
  results.vertices_visited = benchmark_instance.h_vertices_visited[0];
  results.search_depth = benchmark_instance.search_depth;
  results.total_runtime = benchmark_instance.total_runtime;
#endif
  return results;
}

}  // namespace benchmark
}  // namespace gunrock