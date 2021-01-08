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
          typename operator_type,
          typename frontier_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             frontier_type* input,
             frontier_type* output,
             cuda::standard_context_t& __ignore) {
  using vertex_t = typename graph_t::vertex_type;

  // XXX: should use existing context (__ignore)
  mgpu::standard_context_t context(false, __ignore.stream());

  // Get input data of the active buffer.
  auto input_data = input->data();

  // Scan over the work domain to find the output frontier's size.
  auto scanned_work_domain = E->scanned_work_domain;

  auto segment_sizes = [=] __host__ __device__(vertex_t const& v) {
    // if item is invalid, skip processing.
    if (!gunrock::util::limits::is_valid(v))
      return 0;
    auto count = G.get_number_of_neighbors(v);
    return count;  // G.get_number_of_neighbors(v);
  };

  auto new_length = thrust::transform_inclusive_scan(
      thrust::cuda::par.on(__ignore.stream()),  // execution policy
      input->begin(),                           // input iterator: first
      input->end(),                             // input iterator: last
      scanned_work_domain.begin(),              // output iterator
      segment_sizes,                            // unary operation
      thrust::plus<vertex_t>()                  // binary operation
  );

  // The last item contains the total scanned items, so in a simple
  // example, where the input = {1, 0, 2, 2, 1, 3} resulted in the
  // inclusive scan output = {1, 1, 3, 5, 6, 9}, then output.size() - 1
  // will contain the element 9, which is the number of total items to process.
  // We can use this to allocate the size of the output frontier.
  auto location_of_total_scanned_items =
      thrust::distance(scanned_work_domain.begin(), new_length) - 1;

  // Move the last element of the scanned work-domain to host.
  // Last Element = size of active buffer - 1;
  // If the active buffer is greater than number of vertices,
  // we should TODO: resize the scanned work domain, this happens
  // when we allow duplicates to be in the active buffer.
  thrust::host_vector<vertex_t> size_of_output(1, 0);
  cudaMemcpy(size_of_output.data(),
             thrust::raw_pointer_cast(scanned_work_domain.data()) +
                 location_of_total_scanned_items,
             sizeof(vertex_t),  // move one integer
             cudaMemcpyDeviceToHost);

  // If output frontier is empty, resize and return.
  if (!size_of_output[0]) {
    output->resize(size_of_output[0]);
    E->swap_frontier_buffers();
    return;
  }

  // Resize the output (inactive) buffer to the new size.
  output->resize(size_of_output[0]);
  auto output_data = output->data();

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

  mgpu::transform_lbs(neighbors_expand, size_of_output[0],
                      thrust::raw_pointer_cast(scanned_work_domain.data()),
                      (int)input->size(), context);

  // Swap frontier buffers, output buffer now becomes the input buffer and
  // vice-versa.
  E->swap_frontier_buffers();
}

template <advance_type_t type,
          advance_direction_t direction,
          typename graph_t,
          typename enactor_type,
          typename operator_type,
          typename frontier_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             frontier_type* input,
             frontier_type* output,
             cuda::standard_context_t& __ignore) {
  if ((direction == advance_direction_t::forward) ||
      direction == advance_direction_t::backward) {
    execute<type>(G, E, op, input, output, __ignore);
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