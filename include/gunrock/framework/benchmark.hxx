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
        number_of_iterations(0),
        time_per_iteration(),
        real_runtime_collected(false) {}

  thrust::device_vector<unsigned int> edges_visited;
  thrust::device_vector<unsigned int> vertices_visited;

  thrust::host_vector<unsigned int> h_edges_visited;
  thrust::host_vector<unsigned int> h_vertices_visited;

  std::size_t search_depth;
  std::size_t number_of_iterations;  // usually the same as search_depth.
  std::vector<double> time_per_iteration;
  double total_runtime;
  bool real_runtime_collected;
};

struct device_benchmark_t {
  unsigned int* d_edges_visited;
  unsigned int* d_vertices_visited;
};

benchmark_t ____;
__managed__ device_benchmark_t BXXX;

__device__ void LOG_EDGE_VISITED() {
  math::atomic::add(BXXX.d_edges_visited, static_cast<unsigned int>(1));
}

void INIT_BENCH() {
  BXXX.d_edges_visited = ____.edges_visited.data().get();
  BXXX.d_vertices_visited = ____.vertices_visited.data().get();
}

void DESTROY_BENCH() {
  BXXX.d_edges_visited = nullptr;
  BXXX.d_vertices_visited = nullptr;
}

}  // namespace benchmark
}  // namespace gunrock