#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace filter {
namespace predicated {
template <typename graph_t, typename enactor_type, typename operator_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t& context) {
  auto active_buffer = E->get_active_frontier_buffer();
  auto inactive_buffer = E->get_inactive_frontier_buffer();

  using type_t = std::remove_pointer_t<decltype(active_buffer->data())>;

  // Allocate output size.
  inactive_buffer->resize(active_buffer->size());

  auto predicate = [=] __host__ __device__(type_t const& i) -> bool {
    return gunrock::util::limits::is_valid(i) ? op(i) : false;
  };

  // Copy w/ predicate!
  auto new_length = thrust::copy_if(
      thrust::cuda::par.on(context.stream()),         // execution policy
      active_buffer->data(),                          // input iterator: begin
      active_buffer->data() + active_buffer->size(),  // input iterator: end
      inactive_buffer->data(),                        // output iterator
      predicate                                       // predicate
  );

  // XXX: yikes, idk if this is a good idea.
  inactive_buffer->resize((new_length - inactive_buffer->data()));

  //   // Uniquify!
  //   auto new_end = thrust::unique(
  //       thrust::cuda::par.on(context.stream()),  // execution policy
  //       inactive_buffer->data(),                 // input iterator: begin
  //       inactive_buffer->data()
  //       + inactive_buffer->size()  // input iterator: end
  //   );

  //   inactive_buffer->resize((new_end - inactive_buffer->data()));

  E->swap_frontier_buffers();
}
}  // namespace predicated
}  // namespace filter
}  // namespace operators
}  // namespace gunrock