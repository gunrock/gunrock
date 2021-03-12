#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace bypass {
template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             cuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;

  // ... resize as needed.
  if ((output->data() != input->data()) ||
      (output->get_capacity() < input->get_number_of_elements())) {
    output->resize(input->get_number_of_elements());
  }

  output->set_number_of_elements(input->get_number_of_elements());

  // Mark items as invalid instead of removing them (therefore, a "bypass").
  auto bypass = [=] __device__(type_t const& i) {
    if (!gunrock::util::limits::is_valid(i))
      return gunrock::numeric_limits<type_t>::invalid();  // exit early
    return (op(i) ? i : gunrock::numeric_limits<type_t>::invalid());
  };

  // Filter with bypass
  thrust::transform(thrust::cuda::par.on(context.stream()),  // execution policy
                    input->begin(),   // input iterator: begin
                    input->end(),     // input iterator: end
                    output->begin(),  // output iterator
                    bypass            // predicate
  );
}

template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             cuda::standard_context_t& context) {
  // in-place bypass filter (doesn't require an output frontier.)
  execute(G, op, input, input, context);
}

}  // namespace bypass
}  // namespace filter
}  // namespace operators
}  // namespace gunrock