#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace predicated {
template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // Allocate output size if necessary.
  if (output->size() != input->size()) {
    output->resize(input->size());
  }

  auto predicate = [=] __host__ __device__(type_t const& i) -> bool {
    return gunrock::util::limits::is_valid(i) ? op(i) : false;
  };

  // Copy w/ predicate!
  auto new_length = thrust::copy_if(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: begin
      input->end(),                            // input iterator: end
      output->begin(),                         // output iterator
      predicate                                // predicate
  );

  auto new_size = thrust::distance(output->begin(), new_length);
  output->resize(new_size);

  // Uniquify!
  // auto new_end = thrust::unique(
  //     thrust::cuda::par.on(context.stream()),  // execution policy
  //     output->begin(),                         // input iterator: begin
  //     output->end()                            // input iterator: end
  // );

  // new_size = thrust::distance(output->begin(), new_end);
  // output->resize(new_size);
}
}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock