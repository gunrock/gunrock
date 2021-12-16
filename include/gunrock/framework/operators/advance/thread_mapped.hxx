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
#include <gunrock/cuda/launch_box.hxx>
#include <gunrock/cuda/global.hxx>

#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/for/for.hxx>

namespace gunrock {
namespace operators {
namespace advance {
namespace thread_mapped {

template <typename lambda_t>
void __global__ thread_mapped_kernel(lambda_t neighbors_expand,
                                     std::size_t num_elements) {
  auto idx = cuda::thread::global::id::x();
  if (idx < num_elements)
    neighbors_expand(idx);
}

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
             cuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;

  if (output_type != advance_io_type_t::none) {
    auto size_of_output = compute_output_length(G, &input, segments, context);

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

  auto neighbors_expand = [=] __device__(std::size_t const& idx) {
    auto v = (input_type == advance_io_type_t::graph)
                 ? type_t(idx)
                 : input.get_element_at(idx);

    if (!gunrock::util::limits::is_valid(v))
      return;

    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);

    // #pragma unroll
    for (auto i = 0; i < total_edges; ++i) {
      auto e = i + starting_edge;            // edge id
      auto n = G.get_destination_vertex(e);  // neighbor id
      auto w = G.get_edge_weight(e);         // weight
      bool cond = op(v, n, e, w);

      if (output_type != advance_io_type_t::none) {
        std::size_t out_idx = segments_ptr[idx] + i;
        type_t element =
            (cond && n != v) ? n : gunrock::numeric_limits<type_t>::invalid();
        output.set_element_at(element, out_idx);
      }
    }
  };

  std::size_t num_elements = (input_type == advance_io_type_t::graph)
                                 ? G.get_number_of_vertices()
                                 : input.get_number_of_elements();

  using namespace gunrock::cuda::launch_box;
  using launch_t = launch_box_t<launch_params_t<fallback, dim3_t<128>>>;

  dim3 grid_dimensions =
      dim3((num_elements + launch_t::block_dimensions::x - 1) /
               launch_t::block_dimensions::x,
           1, 1);

  // Launch blocked-mapped advance kernel.
  thread_mapped_kernel<<<grid_dimensions,  // grid dimensions
                         launch_t::block_dimensions::get_dim3(),  // block
                                                                  // dimensions
                         launch_t::shared_memory_bytes,  // shared memory
                         context.stream()                // context
                         >>>(neighbors_expand, num_elements);
  context.synchronize();
}
}  // namespace thread_mapped
}  // namespace advance
}  // namespace operators
}  // namespace gunrock