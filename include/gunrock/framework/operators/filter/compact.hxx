#pragma once

#include <moderngpu/kernel_compact.hxx>
#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace compact {

template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;

  using size_type = decltype(input->size());

  auto compact = mgpu::transform_compact(input->size(), *(context.mgpu()));
  auto input_data = input->data();
  int stream_count = compact.upsweep([=] __device__(size_type idx) {
    auto item = input_data[idx];
    // if input item is valid, process the lambda, otherwise return false.
    return gunrock::util::limits::is_valid(item) ? op(item) : false;
  });

  output->resize(stream_count);

  auto output_data = output->data();
  compact.downsweep([=] __device__(size_type dest_idx, size_type source_idx) {
    output_data[dest_idx] = input_data[source_idx];
  });
}
}  // namespace compact
}  // namespace filter
}  // namespace operators
}  // namespace gunrock