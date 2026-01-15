#pragma once

// #include <moderngpu/kernel_compact.hxx>
#include <gunrock/error.hxx>
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
             gcuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;
  using size_type = decltype(input->get_number_of_elements());

  // ModernGPU support has been removed. This operator is no longer available.
  error::throw_if_exception(hipErrorUnknown,
                            "compact filter operator is no longer supported "
                            "due to ModernGPU removal.");
}
}  // namespace compact
}  // namespace filter
}  // namespace operators
}  // namespace gunrock
