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

#include <thrust/transform_scan.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace merge_path {
template <advance_type_t type,
          typename graph_t,
          typename enactor_type,
          typename operator_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& __ignore) {
  using vertex_t = typename graph_t::vertex_type;

  // XXX: should use existing context (__ignore)
  mgpu::standard_context_t context(false, __ignore.stream());

  // Used as an input buffer (frontier)
  auto active_buffer = E->get_active_frontier_buffer();
  // Used as an output buffer (frontier)
  auto inactive_buffer = E->get_inactive_frontier_buffer();

  // Get input data of the active buffer.
  auto input_data = active_buffer->data();

  // Scan over the work domain to find the output frontier's size.
  auto scanned_work_domain = E->scanned_work_domain.data().get();

  auto segment_sizes = [=] __host__ __device__(vertex_t const& v) {
    // if item is invalid, skip processing.
    if (!gunrock::util::limits::is_valid(v))
      return 0;
    return G.get_number_of_neighbors(v);
  };

  auto new_length = thrust::transform_inclusive_scan(
      thrust::cuda::par.on(__ignore.stream()),  // execution policy
      input_data,                               // input iterator: first
      input_data + active_buffer->size(),       // input iterator: last
      scanned_work_domain,                      // output iterator
      segment_sizes,                            // unary operation
      thrust::plus<vertex_t>()                  // binary operation
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
    output_data[idx] =
        cond ? n : gunrock::numeric_limits<decltype(v)>::invalid();
  };

  mgpu::transform_lbs(neighbors_expand, size_of_output[0], scanned_work_domain,
                      (int)active_buffer->size(), context);

  // Swap frontier buffers, output buffer now becomes the input buffer and
  // vice-versa.
  E->swap_frontier_buffers();
}

template <advance_type_t type,
          advance_direction_t direction,
          typename graph_t,
          typename enactor_type,
          typename operator_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& __ignore) {
  if ((direction == advance_direction_t::forward) ||
      direction == advance_direction_t::backward) {
    execute<type>(G, E, op, __ignore);
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