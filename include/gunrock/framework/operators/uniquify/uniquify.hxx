#pragma once

#include <gunrock/cuda/context.hxx>
#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/util/type_limits.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/framework/operators/uniquify/unique.hxx>
#include <gunrock/framework/operators/uniquify/unique_copy.hxx>

namespace gunrock {
namespace operators {
namespace uniquify {

template <uniquify_algorithm_t type, typename frontier_t>
void execute(frontier_t* input,
             frontier_t* output,
             gcuda::multi_context_t& context,
             bool best_effort_uniquification = false,
             const float uniquification_percent = 100) {
  if (context.size() == 1) {
    auto single_context = context.get_context(0);

    if (type == uniquify_algorithm_t::unique) {
      if (!best_effort_uniquification && (uniquification_percent == 100))
        input->sort(sort::order_t::ascending, single_context->stream());
      unique::execute(input, output, *single_context);
    } else if (type == uniquify_algorithm_t::unique_copy) {
      if (!best_effort_uniquification && (uniquification_percent == 100))
        input->sort(sort::order_t::ascending, single_context->stream());
      unique_copy::execute(input, output, *single_context);
    } else {
      error::throw_if_exception(cudaErrorUnknown, "Unqiue type not supported.");
    }
  }

  // Multi-GPU not supported.
  else {
    error::throw_if_exception(cudaErrorUnknown,
                              "`context.size() != 1` not supported");
  }
}

template <uniquify_algorithm_t type = uniquify_algorithm_t::unique,
          typename enactor_type>
void execute(enactor_type* E,
             gcuda::multi_context_t& context,
             bool best_effort_uniquification = false,
             const float uniquification_percent = 100,
             bool swap_buffers = true) {
  if (!best_effort_uniquification)
    if (uniquification_percent < 0 || uniquification_percent > 100)
      error::throw_if_exception(
          cudaErrorUnknown,
          "Uniquification percentage must be a +ve float between 0 and 100.");

  execute<type>(E->get_input_frontier(),     // input frontier
                E->get_output_frontier(),    // output frontier
                context,                     // context
                best_effort_uniquification,  // best effort attempt
                uniquification_percent       // percentage in float
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

}  // namespace uniquify
}  // namespace operators
}  // namespace gunrock