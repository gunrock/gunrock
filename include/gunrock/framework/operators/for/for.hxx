#pragma once

#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace gunrock {
namespace operators {
namespace parallel_for {

template <parallel_for_each_t type, typename graph_t, typename operator_t>
void execute(graph_t& G, operator_t op, cuda::multi_context_t& context) {
  using index_t = std::conditional_t<type == parallel_for_each_t::vertex,
                                     typename graph_t::vertex_type,
                                     typename graph_t::edge_type>;

  auto single_context = context.get_context(0);

  std::size_t size = (type == parallel_for_each_t::vertex)
                         ? G.get_number_of_vertices()
                         : G.get_number_of_edges();

  if constexpr (type == parallel_for_each_t::weight) {
    auto apply = [=] __device__(index_t const& x) { op(G.get_edge_weight(x)); };
    thrust::for_each(
        thrust::cuda::par.on(single_context->stream()),
        thrust::make_counting_iterator<index_t>(0),     // Begin: 0
        thrust::make_counting_iterator<index_t>(size),  // End: # of V/E
        apply                                           // Unary Operator
    );
  } else {
    auto apply = [=] __device__(index_t const& x) { op(x); };
    thrust::for_each(
        thrust::cuda::par.on(single_context->stream()),
        thrust::make_counting_iterator<index_t>(0),     // Begin: 0
        thrust::make_counting_iterator<index_t>(size),  // End: # of V/E
        apply                                           // Unary Operator
    );
  }
}

}  // namespace parallel_for
}  // namespace operators
}  // namespace gunrock