/**
 * @file all_edges.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-01-12
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/transform_scan.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace all_edges {

template <typename graph_t, typename operator_t, typename type_t>
__global__ void all_edges_kernel(const graph_t G,
                                 operator_t op,
                                 type_t* output) {
  auto e = blockIdx.x * blockDim.x + threadIdx.x;
  while (e - threadIdx.x < G.get_number_of_edges()) {
    if (e < G.get_number_of_edges()) {
      auto pair = G.get_source_and_destination_vertices(e);
      auto w = G.get_edge_weight(e);
      bool cond = op(pair.source, pair.destination, e, w);
      output[e] =
          cond ? pair.destination : gunrock::numeric_limits<type_t>::invalid();
    }
  }
}

template <advance_type_t type,
          advance_direction_t direction,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  // Prepare output for all edges advance.
  if (G.get_number_of_edges() <= 0) {
    output->resize(0);
    return;
  }

  if (output->size() != G.get_number_of_edges()) {
    output->resize(G.get_number_of_edges());
  }

  // Get the data buffer from the output frontier.
  auto output_data = output->data();

  // TODO: use launch box instead.
  constexpr int BLOCK_SIZE = 256;
  int GRIDE_SIZE = (G.get_number_of_edges() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  all_edges_kernel<<<GRIDE_SIZE, BLOCK_SIZE, 0, context.stream()>>>(
      G, op, output_data);
}
}  // namespace all_edges
}  // namespace advance
}  // namespace operators
}  // namespace gunrock