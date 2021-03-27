/**
 * @file thread_mapped.hxx
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
#include <thrust/iterator/discard_iterator.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace thread_mapped {

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
  using type_t = typename frontier_t::type_t;

  auto size_of_output = compute_output_length(G, input, segments, context);

  // If output frontier is empty, resize and return.
  if (size_of_output <= 0) {
    output->set_number_of_elements(0);
    return;
  }

  // <todo> Resize the output (inactive) buffer to the new size.
  // Can be hidden within the frontier struct.
  if (output->get_capacity() < size_of_output)
    output->reserve(size_of_output);
  output->set_number_of_elements(size_of_output);
  // output->fill(gunrock::numeric_limits<type_t>::invalid());
  // </todo>

  // Get output data of the active buffer.
  auto segments_data = segments.data().get();
  auto input_data = input->data();
  auto output_data = output->data();

  auto pre_condition = [=] __device__(std::size_t const& idx) {
    auto v = input_data[idx];
    return gunrock::util::limits::is_valid(v);
  };

  auto neighbors_expand = [=] __device__(std::size_t const& idx) {
    auto v = input_data[idx];
    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);

    auto offset = segments_data[idx];

    for (auto i = 0; i < total_edges; ++i) {
      auto e = i + starting_edge;            // edge id
      auto n = G.get_destination_vertex(e);  // neighbor id
      auto w = G.get_edge_weight(e);         // weight
      bool cond = op(v, n, e, w);
      output_data[offset + i] =
          (cond && n != v) ? n : gunrock::numeric_limits<type_t>::invalid();
    }

    return gunrock::numeric_limits<type_t>::invalid();
  };

  thrust::transform_if(
      thrust::cuda::par.on(context.stream()),          // execution policy
      thrust::make_counting_iterator<std::size_t>(0),  // input iterator: first
      thrust::make_counting_iterator<std::size_t>(
          input->get_number_of_elements()),  // input iterator: last
      thrust::make_discard_iterator(),       // output iterator: ignore
      neighbors_expand,                      // unary operation
      pre_condition                          // predicate operation
  );
}
}  // namespace thread_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock