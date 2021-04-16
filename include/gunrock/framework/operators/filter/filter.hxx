#pragma once

#include <gunrock/cuda/context.hxx>
#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/util/type_limits.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/framework/operators/filter/compact.hxx>
#include <gunrock/framework/operators/filter/predicated.hxx>
#include <gunrock/framework/operators/filter/bypass.hxx>
#include <gunrock/framework/operators/filter/remove.hxx>

#include <gunrock/framework/operators/uniquify/uniquify.hxx>

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
             cuda::multi_context_t& context,
             bool filter_and_uniquify = true) {
  if (context.size() == 1) {
    auto single_context = context.get_context(0);

    if (type == filter_algorithm_t::compact) {
      compact::execute(G, op, input, output, *single_context);
    } else if (type == filter_algorithm_t::predicated) {
      predicated::execute(G, op, input, output, *single_context);
    } else if (type == filter_algorithm_t::bypass) {
      bypass::execute(G, op, input, output, *single_context);
    } else if (type == filter_algorithm_t::remove) {
      remove::execute(G, op, input, output, *single_context);
    } else {
      error::throw_if_exception(cudaErrorUnknown, "Filter type not supported.");
    }

    /*!
     * @todo Should filter really do uniquify? This is a tedious interface
     * change.
     */
    if (filter_and_uniquify) {
      operators::uniquify::execute<uniquify_algorithm_t::unique>(output, input,
                                                                 context);
      // Simple pointer swap since output is input and vice-versa after the
      // uniquify.
      frontier_t* temp = input;
      input = output;
      output = temp;
      temp = nullptr;
    }
  } else {
    error::throw_if_exception(cudaErrorUnknown,
                              "`context.size() != 1` not supported");
  }
}

template <filter_algorithm_t type,
          typename graph_t,
          typename enactor_type,
          typename operator_t>
void execute(graph_t& G,
             enactor_type* E,
             operator_t op,
             cuda::multi_context_t& context,
             bool filter_and_uniquify = true,
             bool swap_buffers = true) {
  execute<type>(G,                         // graph
                op,                        // operator_t
                E->get_input_frontier(),   // input frontier
                E->get_output_frontier(),  // output frontier
                context,                   // context
                filter_and_uniquify        // flag to deduplicate
  );

  /*!
   * @note if the Enactor interface is used, we, the library writers assume
   * control of the frontiers and swap the input/output buffers as needed,
   * meaning; Swap frontier buffers, output buffer now becomes the input buffer
   * and vice-versa. This can be overridden by `swap_buffers`.
   */
  if (swap_buffers)
    E->swap_frontier_buffers();
}

}  // namespace filter
}  // namespace operators
}  // namespace gunrock