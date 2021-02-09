#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace uniquify {

template <filter_algorithm_t type, typename frontier_t>
void execute(frontier_t* input,
             const float& uniquification_percent,
             cuda::standard_context_t& context) {
  if (uniquification_percent < 0 || uniquification_percent > 100)
    error::throw_if_exception(
        cudaErrorUnknown,
        "Uniquification percentage must be a +ve float between 0 and 100.");

  // Filter algorithm already produces a unique output, or no uniquification
  // needed. TODO: confirm if compact actually generates a unique output.
  if ((type == filter_algorithm_t::compact) || (uniquification_percent == 0))
    return;

  // 100% uniquification; there could be multiple algorithms to perform this, we
  // will stick with thrust for now.
  else if (uniquification_percent == 100) {
    auto new_end = thrust::unique(
        thrust::cuda::par.on(context.stream()),  // execution policy
        input->begin(),                          // input iterator: begin
        input->end()                             // input iterator: end
    );

    auto new_size = thrust::distance(input->begin(), new_end);
    input->resize(new_size);
  }

  else
    error::throw_if_exception(cudaErrorUnknown,
                              "Variable uniquification (less than 100% or "
                              "greater than 0%) is not implemented.");
}

}  // namespace uniquify
}  // namespace filter
}  // namespace operators
}  // namespace gunrock