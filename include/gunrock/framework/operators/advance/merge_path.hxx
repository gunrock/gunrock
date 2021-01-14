/**
 * @file merge_path.hxx
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

// XXX: Replace these later
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

namespace gunrock {
namespace operators {
namespace advance {
namespace merge_path {
template <advance_type_t type,
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
  using vertex_t = typename graph_t::vertex_type;

  // XXX: should use existing context (context)
  mgpu::standard_context_t _context(false, context.stream());

  auto size_of_output_ignore =
      compute_output_length(G, input, segments, context);
  auto size_of_output = compute_output_length(G, input, segments, _context);

  // If output frontier is empty, resize and return.
  if (size_of_output <= 0) {
    output->resize(0);
    return;
  }

  // Resize the output (inactive) buffer to the new size.
  output->resize(size_of_output);
  auto output_data = output->data();

  // Get input data of the active buffer.
  auto input_data = input->data();

  // Expand incoming neighbors, and using a load-balanced transformation
  // (merge-path based load-balancing) run the user defined advance operator on
  // the load-balanced work items.
  auto neighbors_expand = [=] __device__(std::size_t idx, std::size_t seg,
                                         std::size_t rank) {
    auto v = input_data[seg];

    // if item is invalid, skip processing.
    if (!gunrock::util::limits::is_valid(v))
      return;

    auto start_edge = G.get_starting_edge(v);
    auto e = start_edge + rank;
    auto n = G.get_destination_vertex(e);
    auto w = G.get_edge_weight(e);
    bool cond = op(v, n, e, w);
    output_data[idx] = cond ? n : gunrock::numeric_limits<vertex_t>::invalid();
  };

  mgpu::transform_lbs(neighbors_expand, size_of_output,
                      thrust::raw_pointer_cast(segments.data()),
                      (int)input->size(), _context);
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
  if ((direction == advance_direction_t::forward) ||
      direction == advance_direction_t::backward) {
    execute<type>(G, op, input, output, segments, context);
  } else {  // both (forward + backward)

    // Direction-Optimized advance is supported using CSR and CSC graph
    // views/representations. If they are not both present within the
    // \type(graph_t), throw an exception.
    using find_csr_t = typename graph_t::graph_csr_view_t;
    using find_csc_t = typename graph_t::graph_csc_view_t;
    if (!(G.template contains_representation<find_csr_t>() &&
          G.template contains_representation<find_csc_t>())) {
      error::throw_if_exception(cudaErrorUnknown,
                                "CSR and CSC sparse-matrix representations "
                                "required for direction-optimized advance.");
    }
  }
}
}  // namespace merge_path
}  // namespace advance
}  // namespace operators
}  // namespace gunrock