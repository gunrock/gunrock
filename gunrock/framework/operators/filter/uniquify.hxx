#pragma once

#include <moderngpu/kernel_compact.hxx>
#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace uniquify {

template <typename graph_type, typename enactor_type, typename operator_type>
void execute(graph_type* G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& __ignore) {
  // XXX: should use existing context (__ignore)
  mgpu::standard_context_t context(false);
  auto active_buffer = E->get_active_frontier_buffer();
  auto inactive_buffer = E->get_inactive_frontier_buffer();

  auto compact = mgpu::transform_compact(active_buffer->size(), context);
  auto input_data = active_buffer->data();
  int stream_count = compact.upsweep([=] __device__(int idx) {
    auto item = input_data[idx];
    return op(item);
  });
  inactive_buffer->resize(stream_count);
  auto output_data = inactive_buffer->data();
  compact.downsweep([=] __device__(int dest_idx, int source_idx) {
    output_data[dest_idx] = input_data[source_idx];
  });
  E->swap_frontier_buffers();
}
}  // namespace uniquify
}  // namespace filter
}  // namespace operators
}  // namespace gunrock