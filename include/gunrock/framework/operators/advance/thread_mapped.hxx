/**
 * @file thread_mapped.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Advance operator where a vertex/edge is mapped to a thread.
 * @version 0.1
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/cuda.hxx>

#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/for/for.hxx>

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
             frontier_t& output,
             work_tiles_t& segments,
             gcuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;

  if (output_type != advance_io_type_t::none) {
    auto size_of_output = compute_output_offsets(G, &input, segments, context);

    // If output frontier is empty, resize and return.
    if (size_of_output <= 0) {
      output.set_number_of_elements(0);
      return;
    }

    /// Resize the output (inactive) buffer to the new size.
    /// @todo Can be hidden within the frontier struct.
    if (output.get_capacity() < size_of_output)
      output.reserve(size_of_output);
    output.set_number_of_elements(size_of_output);
  }

  // Get output data of the active buffer.
  auto segments_ptr = segments.data().get();

  auto thread_mapped = [=] __device__(int const& tid, int const& bid) {
    auto v = (input_type == advance_io_type_t::graph)
                 ? type_t(tid)
                 : input.get_element_at(tid);

    if (!gunrock::util::limits::is_valid(v))
      return;

    auto total_edges = G.get_number_of_neighbors(v);

    for (auto i = 0; i < total_edges; ++i) {
      auto starting_edge = G.get_starting_edge(v);
      auto e = i + starting_edge;            // edge id
      auto n = G.get_destination_vertex(e);  // neighbor id
      auto w = G.get_edge_weight(e);         // weight
      bool cond = op(v, n, e, w);

      if (output_type != advance_io_type_t::none) {
        std::size_t out_idx = segments_ptr[tid] + i;
        type_t element = cond ? n : gunrock::numeric_limits<type_t>::invalid();
        output.set_element_at(element, out_idx);
      }
    }
  };

  std::size_t num_elements = (input_type == advance_io_type_t::graph)
                                 ? G.get_number_of_vertices()
                                 : input.get_number_of_elements();

  // Set-up and launch thread-mapped advance.
  using namespace gcuda::launch_box;
  using launch_t =
      launch_box_t<launch_params_dynamic_grid_t<fallback, dim3_t<256>, 3>>;

  launch_t l;
  l.launch_blocked(context, thread_mapped, num_elements);
  context.synchronize();
}
}  // namespace thread_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock
