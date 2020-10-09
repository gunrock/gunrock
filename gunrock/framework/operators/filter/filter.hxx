#pragma once

namespace gunrock {
namespace operators {

enum filter_type_t { predicated, uniquify };

namespace filter {

template <filter_type_t type = filter_type_t::predicated,
          typename graph_type,
          typename enactor_type,
          typename operator_type>
void execute(graph_type* G, enactor_type* E, operator_type op) {
  auto active_buffer = E->get_active_frontier_buffer();
  auto inactive_buffer = E->get_inactive_frontier_buffer();
  auto min_frontier_size = active_buffer->size();
  inactive_buffer->resize(min_frontier_size);
  thrust::copy_if(
      thrust::device,                             // execution policy
      active_buffer->data(),                      // input iterator: begin
      active_buffer->data() + min_frontier_size,  // input iterator: end
      inactive_buffer->data(),                    // output iterator
      op                                          // predicate
  );
}

}  // namespace filter
}  // namespace operators
}  // namespace gunrock