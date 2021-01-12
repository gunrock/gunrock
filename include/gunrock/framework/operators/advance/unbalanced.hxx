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
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_type op,
             frontier_type* input,
             frontier_type* output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;

  // Get input data of the active buffer.
  auto input_data = input->data();

  auto segment_sizes = [=] __host__ __device__(vertex_t const& v) {
    // if item is invalid, segment size is 0.
    if (!gunrock::util::limits::is_valid(v))
      return 0;
    return G.get_number_of_neighbors(v);
  };

  auto new_length = thrust::transform_inclusive_scan(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: first
      input->end(),                            // input iterator: last
      segments.begin(),                        // output iterator
      segment_sizes,                           // unary operation
      thrust::plus<vertex_t>()                 // binary operation
  );

  // The last item contains the total scanned items, so in a simple
  // example, where the input = {1, 0, 2, 2, 1, 3} resulted in the
  // inclusive scan output = {1, 1, 3, 5, 6, 9}, then output.size() - 1
  // will contain the element 9, which is the number of total items to process.
  // We can use this to allocate the size of the output frontier.
  auto location_of_total_scanned_items =
      thrust::distance(segments.begin(), new_length) - 1;

  // Move the last element of the scanned work-domain to host.
  // Last Element = size of active buffer - 1;
  // If the active buffer is greater than number of vertices,
  // we should TODO: resize the scanned work domain, this happens
  // when we allow duplicates to be in the active buffer.
  thrust::host_vector<vertex_t> size_of_output(1, 0);
  cudaMemcpy(size_of_output.data(),
             thrust::raw_pointer_cast(segments.data()) +
                 location_of_total_scanned_items,
             sizeof(vertex_t),  // move one integer
             cudaMemcpyDeviceToHost);

  // If output frontier is empty, resize and return.
  if (!size_of_output[0]) {
    output->resize(size_of_output[0]);
    return;
  }

  // Resize the output (inactive) buffer to the new size.
  output->resize(size_of_output[0]);
  auto output_data = output->data();

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
                    input->begin(),   // input iterator: first
                    input->end(),     // input iterator: last
                    input->begin(),   // in-place transform
                    neighbors_expand  // unary operation
  );
}
}  // namespace unbalanced
}  // namespace advance
}  // namespace operators
}  // namespace gunrock