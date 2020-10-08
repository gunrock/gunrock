/**
 * @file advance.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace operators {

enum advance_type_t {
  vertex_to_vertex,
  vertex_to_edge,
  edge_to_edge,
  edge_to_vertex
};

enum advance_direction_t {
  forward,   // Push-based approach
  backward,  // Pull-based approach
  both       // Push-pull optimized
};

namespace advance {

template <typename graph_type,
          typename frontier_buffer_type,
          typename operator_type>
__global__ void simple(graph_type* G,
                       frontier_buffer_type* buffer_a,
                       frontier_buffer_type* buffer_b,
                       std::size_t frontier_size,
                       operator_type op) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= frontier_size)
    return;

  printf("frontier size = %d\n", frontier_size);
  auto v = buffer_a[tid];
  printf("source = %d\n", v);
  auto starting_edge = G->get_starting_edge(v);
  auto total_edges = G->get_number_of_neighbors(v);
  for (auto e = starting_edge; e < total_edges; ++e) {
    auto n = G->get_destination_vertex(e);
    bool valid = op(v, n, e, G->get_edge_weight(e));
    if (valid)
      printf("Add %d to frontier.\n", n);
  }
}

template <advance_type_t type = advance_type_t::vertex_to_vertex,
          typename graph_type,
          typename frontier_buffers_type,
          typename operator_type>
void execute(graph_type* G, frontier_buffers_type* F, operator_type op) {
  simple<<<256, 256>>>(G, F[0].data(), F[1].data(), F[0].get_frontier_size(),
                       op);
  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());
  std::cout << "New size: " << F[1].get_frontier_size() << std::endl;
  F->set_frontier_size(F[1].get_frontier_size());
  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());
}
}  // namespace advance
}  // namespace operators
}  // namespace gunrock