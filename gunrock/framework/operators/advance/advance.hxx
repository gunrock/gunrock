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

#include <bits/stdc++.h>
#include <gunrock/util/math.hxx>

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

template <typename graph_type, typename frontier_type, typename operator_type>
__global__ void simple(graph_type* G,
                       frontier_type* input,
                       frontier_type* output,
                       std::size_t frontier_size,
                       int* output_size,
                       operator_type op) {
  using vertex_t = typename graph_type::vertex_type;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= frontier_size)
    return;

  auto v = input[tid];
  auto starting_edge = G->get_starting_edge(v);
  auto total_edges = G->get_number_of_neighbors(v);
  for (auto e = starting_edge; e < total_edges; ++e) {
    auto n = G->get_destination_vertex(e);
    bool valid = op(v, n, e, G->get_edge_weight(e));
    if (valid) {
      output[e] = n;
      math::atomic::add(output_size, 1);
    } else {
      output[e] = std::numeric_limits<vertex_t>::max();
    }
  }
}

template <advance_type_t type = advance_type_t::vertex_to_vertex,
          typename graph_type,
          typename enactor_type,
          typename operator_type>
void execute(graph_type* G, enactor_type* E, operator_type op) {
  thrust::device_vector<int> output_size(1, 0);

  auto active_buffer =
      E->get_active_frontier_buffer();  // Used as an input buffer
  auto inactive_buffer =
      E->get_inactive_frontier_buffer();  // Used as an output buffer

  // Run a simple advance
  simple<<<256, 256>>>(G, active_buffer->data(), inactive_buffer->data(),
                       active_buffer->size(), output_size.data().get(), op);

  // Swap input/output buffers for future iterations
  thrust::host_vector<int> new_output_size = output_size;
  inactive_buffer->resize(new_output_size[0]);
  E->swap_frontier_buffers();
}
}  // namespace advance
}  // namespace operators
}  // namespace gunrock