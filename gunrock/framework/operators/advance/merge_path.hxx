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
template <advance_type_t type = advance_type_t::vertex_to_vertex,
          advance_direction_t direction = advance_direction_t::forward,
          typename graph_type,
          typename enactor_type,
          typename operator_type>
void execute(graph_type* G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& __ignore) {
  // XXX: should use existing context (__ignore)
  mgpu::standard_context_t context(false);

  // Used as an input buffer (frontier)
  auto active_buffer = E->get_active_frontier_buffer();
  // Used as an output buffer (frontier)
  auto inactive_buffer = E->get_inactive_frontier_buffer();

  // Get input data of the active buffer.
  auto input_data = active_buffer->data();

  // Scan over the work domain to find the output frontier's size.
  auto scanned_work_domain = E->scanned_work_domain.data().get();
  thrust::device_vector<int> count(1, 0);

  auto segment_sizes = [G, input_data] __device__(int idx) {
    int count = 0;
    int v = input_data[idx];
    count = G->get_number_of_neighbors(v);
    return count;
  };

  mgpu::transform_scan<int>(segment_sizes, (int)active_buffer->size(),
                            scanned_work_domain, mgpu::plus_t<int>(),
                            count.data(), context);

  // If output frontier is empty, resize and return.
  thrust::host_vector<int> front = count;
  if (!front[0]) {
    inactive_buffer->resize(front[0]);
    return;
  }

  // Resize the output (inactive) buffer to the new size.
  inactive_buffer->resize(front[0]);
  auto output_data = inactive_buffer->data();

  // Expand incoming neighbors, and using a load-balanced transformation
  // (merge-path based load-balancing) run the user defined advance operator on
  // the load-balanced work items.
  auto neighbors_expand = [G, op, input_data, output_data] __device__(
                              int idx, int seg, int rank) {
    auto v = input_data[seg];
    auto start_edge = G->get_starting_edge(v);
    auto e = start_edge + rank;
    auto n = G->get_destination_vertex(e);
    auto w = G->get_edge_weight(e);
    bool cond = op(v, n, e, w);
    output_data[idx] = cond ? n : std::numeric_limits<decltype(v)>::max();
  };

  mgpu::transform_lbs(neighbors_expand, front[0], scanned_work_domain,
                      (int)active_buffer->size(), context);

  // Swap frontier buffers, output buffer now becomes the input buffer and
  // vice-versa.
  E->swap_frontier_buffers();
}
}  // namespace merge_path
}  // namespace advance
}  // namespace operators
}  // namespace gunrock