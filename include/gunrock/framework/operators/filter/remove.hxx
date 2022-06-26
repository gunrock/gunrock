#pragma once

#include <gunrock/framework/operators/configs.hxx>
#include <thrust/remove.h>

namespace gunrock {
namespace operators {
namespace filter {
namespace remove {
template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             gcuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;

  // Allocate output size if necessary.
  // if (output->get_capacity() != input->get_number_of_elements())
  output->reserve(input->get_number_of_elements());
  output->set_number_of_elements(input->get_number_of_elements());

  auto predicate = [=] __host__ __device__(type_t const& i) -> bool {
    return gunrock::util::limits::is_valid(i) ? !op(i) : true;
  };

  // Copy w/ predicate!
  auto new_length = thrust::remove_copy_if(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: begin
      input->end(),                            // input iterator: end
      output->begin(),                         // output iterator: begin
      predicate                                // predicate
  );

  auto new_size = thrust::distance(output->begin(), new_length);
  output->set_number_of_elements(new_size);
}
}  // namespace remove
}  // namespace filter
}  // namespace operators
}  // namespace gunrock