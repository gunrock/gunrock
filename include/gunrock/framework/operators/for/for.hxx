#pragma once

#include <type_traits>

#include <gunrock/cuda/context.hxx>
#include <gunrock/framework/operators/configs.hxx>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace gunrock {
namespace operators {
namespace parallel_for {

/**
 * @brief For each element in the frontier, apply a user-defined function.
 *
 * @tparam type parallel_for_each_t::element
 * @tparam func_t User-defined function type.
 * @tparam frontier_t Frontier type.
 * @param f Frontiers to apply user-defined function to.
 * @param op User-defined function.
 * @param context Device context (@see gcuda::multi_context_t).
 * @return bool ignore the output, limitation of `__device__` lambda functions
 * require a template parameter to be named (see
 * https://github.com/neoblizz/enable_if_bug).
 */
template <parallel_for_each_t type, typename func_t, typename frontier_t>
std::enable_if_t<type == parallel_for_each_t::element>
execute(frontier_t& f, func_t op, gcuda::multi_context_t& context) {
  static_assert(type == parallel_for_each_t::element);
  using type_t = typename frontier_t::type_t;
  auto single_context = context.get_context(0);
  /// TODO: use get and set frontier elements instead.
  thrust::for_each(thrust::cuda::par.on(single_context->stream()),
                   f.begin(),  // Begin: 0
                   f.end(),    // End: # of V/E
                   [=] __device__(type_t const& x) {
                     if (gunrock::util::limits::is_valid(x))
                       op(x);
                   }  // Unary Operator
  );
}

/**
 * @brief For each vertex, edge or edge weight in the graph, apply a
 * user-defined function.
 *
 * @tparam type parallel_for_each_t::vertex, parallel_for_each_t::edge, or
 * parallel_for_each_t::weight
 * @tparam func_t User-defined function type.
 * @tparam graph_t Graph type.
 * @param G Graph to apply user-defined function to.
 * @param op User-defined function.
 * @param context Device context (@see gcuda::multi_context_t).
 * @return bool ignore the output, limitation of `__device__` lambda functions
 * require a template parameter to be named (see
 * https://github.com/neoblizz/enable_if_bug).
 */
template <parallel_for_each_t type, typename func_t, typename graph_t>
std::enable_if_t<type != parallel_for_each_t::element>
execute(graph_t& G, func_t op, gcuda::multi_context_t& context) {
  static_assert((type == parallel_for_each_t::weight) ||
                (type == parallel_for_each_t::edge) ||
                (type == parallel_for_each_t::vertex));
  using index_t = std::conditional_t<type == parallel_for_each_t::vertex,
                                     typename graph_t::vertex_type,
                                     typename graph_t::edge_type>;
  auto single_context = context.get_context(0);
  std::size_t size = (type == parallel_for_each_t::vertex)
                         ? G.get_number_of_vertices()
                         : G.get_number_of_edges();

  /// Note: For certain host platform/dialect, an extended lambda cannot be
  /// defined inside the 'if' or 'else' block of a constexpr if statement.
  switch (type) {
    case parallel_for_each_t::weight:
      thrust::for_each(
          thrust::cuda::par.on(single_context->stream()),
          thrust::make_counting_iterator<index_t>(0),     // Begin: 0
          thrust::make_counting_iterator<index_t>(size),  // End: # of V/E
          [=] __device__(index_t const& x) {
            op(G.get_edge_weight(x));
          }  // Unary Operator
      );
      break;
    default:
      thrust::for_each(
          thrust::cuda::par.on(single_context->stream()),
          thrust::make_counting_iterator<index_t>(0),     // Begin: 0
          thrust::make_counting_iterator<index_t>(size),  // End: # of V/E
          [=] __device__(index_t const& x) { op(x); }     // Unary Operator
      );
      break;
  }
}

}  // namespace parallel_for
}  // namespace operators
}  // namespace gunrock