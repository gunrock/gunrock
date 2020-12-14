/**
 * @file unbalanced.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/for/for.hxx>

#include <thrust/transform_scan.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace unbalanced {

template <advance_type_t type,
          advance_direction_t direction,
          typename graph_t,
          typename enactor_type,
          typename operator_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;
  // Used as an input buffer (frontier)
  auto active_buffer = E->get_active_frontier_buffer();
  // Used as an output buffer (frontier)
  auto inactive_buffer = E->get_inactive_frontier_buffer();

  // Get input data of the active buffer.
  auto input_data = active_buffer->data();

  // Scan over the work domain to find the output frontier's size.
  auto scanned_work_domain = E->scanned_work_domain.data().get();

  auto segment_sizes = [=] __host__ __device__(vertex_t const& v) {
    // if item is invalid, segment size is 0.
    if (!gunrock::util::limits::is_valid(v))
      return 0;
    return G.get_number_of_neighbors(v);
  };

  auto new_length = thrust::transform_inclusive_scan(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input_data,                              // input iterator: first
      input_data + active_buffer->size(),      // input iterator: last
      scanned_work_domain,                     // output iterator
      segment_sizes,                           // unary operation
      thrust::plus<vertex_t>()                 // binary operation
  );

  // Move the last element of the scanned work-domain to host.
  // Last Element = size of active buffer - 1;
  // If the active buffer is greater than number of vertices,
  // we should TODO: resize the scanned work domain, this happens
  // when we allow duplicates to be in the active buffer.
  thrust::host_vector<vertex_t> size_of_output(1, 0);
  cudaMemcpy(size_of_output.data(),
             scanned_work_domain + active_buffer->size() - 1,
             sizeof(vertex_t),  // move one integer
             cudaMemcpyDeviceToHost);

  // If output frontier is empty, resize and return.
  if (!size_of_output[0]) {
    inactive_buffer->resize(size_of_output[0]);
    E->swap_frontier_buffers();
    return;
  }

  // Resize the output (inactive) buffer to the new size.
  inactive_buffer->resize(size_of_output[0]);
  auto output_data = inactive_buffer->data();

  auto neighbors_expand = [=] __device__(vertex_t const& v) {
    // if item is invalid, skip processing.
    if (!gunrock::util::limits::is_valid(v))
      return gunrock::numeric_limits<vertex_t>::invalid();

    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);
    for (auto e = starting_edge; e < starting_edge + total_edges; ++e) {
      auto n = G.get_destination_vertex(e);
      auto w = G.get_edge_weight(e);
      bool cond = op(v, n, e, w);
      output_data[e] = cond ? n : gunrock::numeric_limits<vertex_t>::invalid();
    }

    return gunrock::numeric_limits<vertex_t>::invalid();
  };

  thrust::transform(thrust::cuda::par.on(context.stream()),  // execution policy
                    input_data,  // input iterator: first
                    input_data + active_buffer->size(),  // input iterator: last
                    input_data,                          // in-place transform
                    neighbors_expand                     // unary operation
  );

  // Swap frontier buffers, output buffer now becomes the input buffer and
  // vice-versa.
  E->swap_frontier_buffers();
}
}  // namespace unbalanced
}  // namespace advance
}  // namespace operators
}  // namespace gunrock