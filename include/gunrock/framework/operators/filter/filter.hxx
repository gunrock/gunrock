#pragma once

#include <gunrock/cuda/context.hxx>
#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/util/type_limits.hxx>
#include <gunrock/util/type_traits.hxx>

#include <gunrock/framework/operators/filter/compact.hxx>
#include <gunrock/framework/operators/filter/predicated.hxx>
#include <gunrock/framework/operators/filter/bypass.hxx>
#include <gunrock/framework/operators/filter/remove.hxx>

#include <gunrock/framework/operators/uniquify/uniquify.hxx>

namespace gunrock {
namespace operators {
namespace filter {

/**
 * @brief A filter operator generates a new frontier from an input frontier by
 * removing elements in the frontier that do not satisfy a predicate.
 *
 * @par Overview
 * A frontier consists of vertices or edges, a filter applied to an incoming
 * frontier will generate a new frontier by removing the elements that do not
 * satisfy a user-defined predicate. The predicate function takes a vertex or
 * edge and returns a boolean value. If the boolean value is `true`, the element
 * is kept in the new frontier. If the boolean value is `false`, the element is
 * removed from the new frontier.
 *
 * @par Example
 * The following code is a simple snippet on how to use filter operator with
 * input and output frontiers.
 *
 * \code {.cpp}
 * auto sample = [=] __host__ __device__(
 *   vertex_t const& vertex) -> bool {
 *   return (vertex % 2 == 0) ? true : false; // keep even vertices
 * };
 *
 * // Execute filter operator on the provided lambda
 * operators::filter::execute<operators::filter_algorithm_t::bypass>(
 *   G, sample, input_frontier, output_frontier, context);
 * \endcode
 *
 *
 * @tparam alg_type The filter algorithm to use (@see filter_algorithm_t).
 * @tparam graph_t The graph type.
 * @tparam operator_t The operator type.
 * @tparam frontier_t The frontier type.
 * @param G Input graph used.
 * @param op Predicate function, can be defined using a C++ lambda function.
 * @param input Input frontier.
 * @param output Output frontier (some algorithms may not use this, and allow
 * for in-place filter operation).
 * @param context a `gcuda::multi_context_t` that contains GPU contexts for the
 * available CUDA devices. Used to launch the filter kernels.
 */
template <filter_algorithm_t alg_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             gcuda::multi_context_t& context) {
  if (context.size() == 1) {
    auto single_context = context.get_context(0);

    if constexpr (alg_type == filter_algorithm_t::compact) {
      compact::execute(G, op, input, output, *single_context);
    } else if (alg_type == filter_algorithm_t::predicated) {
      predicated::execute(G, op, input, output, *single_context);
    } else if (alg_type == filter_algorithm_t::bypass) {
      bypass::execute(G, op, input, output, *single_context);
    } else if (alg_type == filter_algorithm_t::remove) {
      remove::execute(G, op, input, output, *single_context);
    } else {
      error::throw_if_exception(cudaErrorUnknown, "Filter type not supported.");
    }
  } else {
    error::throw_if_exception(cudaErrorUnknown,
                              "`context.size() != 1` not supported");
  }
}

/**
 * @brief A filter operator generates a new frontier from an input frontier by
 * removing elements in the frontier that do not satisfy a predicate.
 *
 * @par Overview
 * A frontier consists of vertices or edges, a filter applied to an incoming
 * frontier will generate a new frontier by removing the elements that do not
 * satisfy a user-defined predicate. The predicate function takes a vertex or
 * edge and returns a boolean value. If the boolean value is `true`, the element
 * is kept in the new frontier. If the boolean value is `false`, the element is
 * removed from the new frontier.
 *
 * @par Example
 * The following code is a simple snippet on how to use filter within the
 * enactor loop instead of explicit input and output frontiers. The enactor
 * interface automatically swap the input and output after each iteration.
 *
 * \code {.cpp}
 * auto sample = [=] __host__ __device__(
 *   vertex_t const& vertex) -> bool {
 *   return (vertex % 2 == 0) ? true : false; // keep even vertices
 * };
 *
 * // Execute filter operator on the provided lambda
 * operators::filter::execute<operators::filter_algorithm_t::bypass>(
 *   G, E, sample, context);
 * \endcode
 *
 *
 * @tparam alg_type The filter algorithm to use (@see filter_algorithm_t).
 * @tparam graph_t The graph type.
 * @tparam enactor_type The enactor type.
 * @tparam operator_t The operator type.
 * @tparam frontier_t The frontier type.
 * @param G Input graph used.
 * @param op Predicate function, can be defined using a C++ lambda function.
 * @param E Enactor struct containing input and output frontiers.
 * @param context a `gcuda::multi_context_t` that contains GPU contexts for the
 * available CUDA devices. Used to launch the filter kernels.
 */
template <filter_algorithm_t alg_type,
          typename graph_t,
          typename enactor_type,
          typename operator_t>
void execute(graph_t& G,
             enactor_type* E,
             operator_t op,
             gcuda::multi_context_t& context,
             bool swap_buffers = true) {
  execute<alg_type>(G,                         // graph
                    op,                        // operator_t
                    E->get_input_frontier(),   // input frontier
                    E->get_output_frontier(),  // output frontier
                    context                    // context
  );

  /*!
   * @note if the Enactor interface is used, we, the library writers assume
   * control of the frontiers and swap the input/output buffers as needed,
   * meaning; Swap frontier buffers, output buffer now becomes the input buffer
   * and vice-versa. This can be overridden by `swap_buffers`.
   */
  if (swap_buffers)
    E->swap_frontier_buffers();
}

}  // namespace filter
}  // namespace operators
}  // namespace gunrock