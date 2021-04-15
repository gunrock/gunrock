#pragma once

#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace gunrock {
namespace operators {
namespace parallel_for {

template <parallel_for_each_t type, typename graph_t, typename operator_t>
void execute(graph_t& G, operator_t op, cuda::multi_context_t& context) {
  using type_t = std::conditional_t<type == parallel_for_each_t::vertex,
                                    typename graph_t::vertex_type,
                                    typename graph_t::edge_type>;

  auto single_context = context.get_context(0);

  std::size_t size = (type == parallel_for_each_t::vertex)
                         ? G.get_number_of_vertices()
                         : G.get_number_of_edges();
  auto apply = [=] __device__(type_t const& x) {
    op(x);
    return x;  // output ignored.
  };

  thrust::transform(
      thrust::cuda::par.on(single_context->stream()),
      thrust::make_counting_iterator<type_t>(0),     // Begin: 0
      thrust::make_counting_iterator<type_t>(size),  // End: # of Edges
      thrust::make_discard_iterator(),               // output iterator: ignore
      apply                                          // Unary Operator
  );
}

}  // namespace parallel_for
}  // namespace operators
}  // namespace gunrock