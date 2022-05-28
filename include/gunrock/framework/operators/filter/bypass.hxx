#pragma once

#include <gunrock/framework/operators/configs.hxx>
#include <thrust/transform.h>

namespace gunrock {
namespace operators {
namespace filter {
namespace bypass {
template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             gcuda::standard_context_t& context) {
  using type_t = typename frontier_t::type_t;

  // ... resize as needed.
  if ((output->data() != input->data()) ||
      (output->get_capacity() < input->get_number_of_elements())) {
    output->reserve(input->get_number_of_elements());
  }

  output->set_number_of_elements(input->get_number_of_elements());

  auto input_ptr = input->data();

  // Mark items as invalid instead of removing them (therefore, a "bypass").
  auto bypass = [=] __device__(std::size_t const& idx) {
    auto v = input_ptr[idx];
    if (!gunrock::util::limits::is_valid(v))
      return gunrock::numeric_limits<type_t>::invalid();  // exit early
    return (op(v) ? v : gunrock::numeric_limits<type_t>::invalid());
  };

  std::size_t end = input->get_number_of_elements();

  // Filter with bypass
  thrust::transform(
      thrust::cuda::par.on(context.stream()),          // execution policy
      thrust::make_counting_iterator<std::size_t>(0),  // input iterator: first
      thrust::make_counting_iterator<std::size_t>(end),  // input iterator: last
      output->begin(),                                   // output iterator
      bypass                                             // predicate
  );
}

template <typename graph_t, typename operator_t, typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             gcuda::standard_context_t& context) {
  // in-place bypass filter (doesn't require an output frontier.)
  execute(G, op, input, input, context);
}

}  // namespace bypass
}  // namespace filter
}  // namespace operators
}  // namespace gunrock