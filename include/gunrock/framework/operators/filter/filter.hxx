#pragma once

#include <gunrock/cuda/context.hxx>
#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/util/type_limits.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/framework/operators/filter/compact.hxx>
#include <gunrock/framework/operators/filter/predicated.hxx>
#include <gunrock/framework/operators/filter/bypass.hxx>

namespace gunrock {
namespace operators {
namespace filter {

template <filter_algorithm_t type,
          typename graph_t,
          typename operator_t,
          typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t* context) {
  if (type == filter_algorithm_t::compact)
    compact::execute(G, op, input, output, *context);
  else if (type == filter_algorithm_t::predicated)
    predicated::execute(G, op, input, output, *context);
  else if (type == filter_algorithm_t::bypass)
    bypass::execute(G, op, input, output, *context);
  else
    error::throw_if_exception(cudaErrorUnknown, "Filter type not supported.");
}

template <filter_algorithm_t type,
          typename graph_t,
          typename enactor_type,
          typename operator_t>
void execute(graph_t& G,
             enactor_type* E,
             operator_t op,
             cuda::standard_context_t* context) {
  execute<type>(G,                         // graph
                op,                        // operator_t
                E->get_input_frontier(),   // input frontier
                E->get_output_frontier(),  // output frontier
                context                    // context
  );

  E->swap_frontier_buffers();
}

}  // namespace filter
}  // namespace operators
}  // namespace gunrock