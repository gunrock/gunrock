#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace bypass {
template <typename graph_t,
          typename enactor_type,
          typename operator_type,
          typename frontier_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             frontier_type* input,
             frontier_type* output,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;

  auto bypass = [=] __device__(vertex_t const& v) {
    return (op(v) ? v : gunrock::numeric_limits<vertex_t>::invalid());
  };

  // Filter with bypass
  thrust::transform(thrust::cuda::par.on(context.stream()),  // execution policy
                    input->begin(),  // input iterator: begin
                    input->end(),    // input iterator: end
                    input->begin(),  // output iterator
                    bypass           // predicate
  );
}
}  // namespace bypass
}  // namespace filter
}  // namespace operators
}  // namespace gunrock