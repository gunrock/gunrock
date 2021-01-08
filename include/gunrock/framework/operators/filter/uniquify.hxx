#pragma once

#include <moderngpu/kernel_compact.hxx>
#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace uniquify {

template <typename graph_t,
          typename enactor_type,
          typename operator_type,
          typename frontier_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             frontier_type* input,
             frontier_type* output,
             cuda::standard_context_t& __ignore) {
  // XXX: should use existing context (__ignore)
  mgpu::standard_context_t context(false);

  using size_type = decltype(input->size());

  auto compact = mgpu::transform_compact(input->size(), context);
  auto input_data = input->data();
  int stream_count = compact.upsweep([=] __device__(size_type idx) {
    auto item = input_data[idx];
    return gunrock::util::limits::is_valid(item) ? op(item) : false;
  });
  output->resize(stream_count);
  auto output_data = output->data();
  compact.downsweep([=] __device__(size_type dest_idx, size_type source_idx) {
    output_data[dest_idx] = input_data[source_idx];
  });
  E->swap_frontier_buffers();
}
}  // namespace uniquify
}  // namespace filter
}  // namespace operators
}  // namespace gunrock