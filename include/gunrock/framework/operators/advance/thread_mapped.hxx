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

template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t& input,
             frontier_t* output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;
  frontier_storage_t underlying_t = input.get_frontier_storage_t();

  if (output_type != advance_io_type_t::none ||
      underlying_t == frontier_storage_t::boolmap) {
    auto size_of_output = compute_output_length(G, &input, segments, context);

    // If output frontier is empty, resize and return.
    if (size_of_output <= 0) {
      output->set_number_of_elements(0);
      return;
    }

    /// Resize the output (inactive) buffer to the new size.
    /// @todo Can be hidden within the frontier struct.
    if (output->get_capacity() < size_of_output)
      output->reserve(size_of_output);
    output->set_number_of_elements(size_of_output);
  }

  // Get output data of the active buffer.
  auto segments_ptr = segments.data().get();
  auto input_ptr = input.data();
  auto output_ptr = output->data();

  /// @note Pre-Condition is causing two reads,
  /// when only one is required.

  // auto pre_condition = [=] __device__(std::size_t const& idx) {
  //   auto v = (input_type == advance_io_type_t::graph)
  //                ? type_t(idx)
  //                : frontier::get_element_at(idx, input_ptr);

  //   return gunrock::util::limits::is_valid(v);
  // };

  auto neighbors_expand = [=] __device__(std::size_t const& idx) {
    auto v = (input_type == advance_io_type_t::graph)
                 ? type_t(idx)
                 : frontier::get_element_at(idx, input_ptr);

    printf("%p\n", (void*)input.get());

    if (!gunrock::util::limits::is_valid(v))
      return gunrock::numeric_limits<type_t>::invalid();

    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);

    // #pragma unroll
    for (auto i = 0; i < total_edges; ++i) {
      auto e = i + starting_edge;            // edge id
      auto n = G.get_destination_vertex(e);  // neighbor id
      auto w = G.get_edge_weight(e);         // weight
      bool cond = op(v, n, e, w);

      if constexpr (output_type != advance_io_type_t::none) {
        std::size_t out_idx = segments_ptr[idx] + i;
        cond = (cond && n != v);
        type_t element = cond ? n : gunrock::numeric_limits<type_t>::invalid();
        if (underlying_t == frontier_storage_t::boolmap) {
          if (cond)
            frontier::set_element_at<frontier_storage_t::boolmap>(
                (std::size_t)n, element, output_ptr);
        } else
          frontier::set_element_at(out_idx, element, output_ptr);
      }
    }

    return gunrock::numeric_limits<type_t>::invalid();
  };

  std::size_t end = (input_type == advance_io_type_t::graph ||
                     underlying_t == frontier_storage_t::boolmap)
                        ? G.get_number_of_vertices()
                        : input.get_number_of_elements();
  thrust::transform(
      thrust::cuda::par.on(context.stream()),          // execution policy
      thrust::make_counting_iterator<std::size_t>(0),  // input iterator: first
      thrust::make_counting_iterator<std::size_t>(end),  // input iterator: last
      thrust::make_discard_iterator(),  // output iterator: ignore
      neighbors_expand                  // unary operation
      // pre_condition                     // predicate operation
  );

  // reset the input frontier.
  if (underlying_t == frontier_storage_t::boolmap)
    input.fill(0, context.stream());
}
}  // namespace thread_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock