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
template <advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             gcuda::standard_context_t& context) {
  if constexpr (direction == advance_direction_t::optimized) {
    error::throw_if_exception(cudaErrorUnknown,
                              "Direction-optimized not yet implemented.");

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

  using type_t = typename frontier_t::type_t;
  using view_t = std::conditional_t<direction == advance_direction_t::forward,
                                    typename graph_t::graph_csr_view_t,
                                    typename graph_t::graph_csc_view_t>;

  auto size_of_output = compute_output_offsets(
      G, input, segments, context,
      (input_type == advance_io_type_t::graph) ? true : false);

  if constexpr (output_type != advance_io_type_t::none) {
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
    // </todo>
  }

  // Get input/output data of the active buffer.
  auto input_data = input->data();
  auto output_data = output->data();

  // Expand incoming neighbors, and using a load-balanced transformation
  // (merge-path based load-balancing) run the user defined advance operator
  // on the load-balanced work items.
  auto neighbors_expand = [=] __device__(std::size_t idx, std::size_t seg,
                                         std::size_t rank) {
    auto v = (input_type == advance_io_type_t::graph) ? type_t(seg)
                                                      : input_data[seg];

    // if item is invalid, skip processing.
    if (!gunrock::util::limits::is_valid(v))
      return;

    auto start_edge = G.template get_starting_edge<view_t>(v);
    auto e = start_edge + rank;
    auto n = G.template get_destination_vertex<view_t>(e);
    auto w = G.template get_edge_weight<view_t>(e);
    bool cond = op(v, n, e, w);

    if (output_type != advance_io_type_t::none)
      output_data[idx] = cond ? n : gunrock::numeric_limits<type_t>::invalid();
  };

  int end = (input_type == advance_io_type_t::graph)
                ? G.get_number_of_vertices()
                : input->get_number_of_elements();
  mgpu::transform_lbs(neighbors_expand, size_of_output,
                      thrust::raw_pointer_cast(segments.data()), end,
                      *(context.mgpu()));
}
}  // namespace merge_path
}  // namespace advance
}  // namespace operators
}  // namespace gunrock