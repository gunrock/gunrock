#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace predicated {
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
  using type_t = std::remove_pointer_t<decltype(input->data())>;

  // Allocate output size.
  output->resize(input->size());

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

  // // Uniquify!
  // auto new_end = thrust::unique(
  //     thrust::cuda::par.on(context.stream()),  // execution policy
  //     output->begin(),                         // input iterator: begin
  //     output->end()                            // input iterator: end
  // );

  //   output->resize((new_end - output->data()));

  E->swap_frontier_buffers();
}
}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock