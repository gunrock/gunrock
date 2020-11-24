#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace bypass {
template <typename graph_type, typename enactor_type, typename operator_type>
void execute(graph_type* G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_type::vertex_type;
  auto active_buffer = E->get_active_frontier_buffer();

  auto bypass = [op] __device__(vertex_t const& v) {
    return (op(v) ? v : gunrock::numeric_limits<vertex_t>::invalid());
  };

  // Filter with bypass
  thrust::transform(
      thrust::cuda::par.on(context.stream()),         // execution policy
      active_buffer->data(),                          // input iterator: begin
      active_buffer->data() + active_buffer->size(),  // input iterator: end
      active_buffer->data(),                          // output iterator
      bypass                                          // predicate
  );
}
}  // namespace bypass
}  // namespace filter
}  // namespace operators
}  // namespace gunrock