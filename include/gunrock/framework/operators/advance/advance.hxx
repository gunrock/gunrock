/**
 * @file advance.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Advance operator's header-file.
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/cuda/cuda.hxx>
#include <gunrock/error.hxx>
#include <gunrock/util/type_limits.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <gunrock/framework/operators/advance/helpers.hxx>

#include <gunrock/framework/operators/advance/thread_mapped.hxx>
#include <gunrock/framework/operators/advance/block_mapped.hxx>
#include <gunrock/framework/operators/advance/bucketing.hxx>
#include <gunrock/framework/operators/advance/merge_path.hxx>

#if __HIP_PLATFORM_NVIDIA__
#include <gunrock/framework/operators/advance/merge_path_v2.hxx>
#endif

namespace gunrock {
namespace operators {
namespace advance {

/**
 * @brief An advance operator generates a new frontier from an input frontier
 * by visiting the neighbors of the input frontier.
 *
 * @par Overview
 * A frontier can consist of either vertices or edges, and an advance step can
 * input and output either kind of frontier. The output frontier only contains
 * the neighbors and not the source vertex/edges of the input frontier. Advance
 * is an irregularly-parallel operation for two reasons:
 *  1. different vertices in a graph have different numbers of neighbors and
 *  2. vertices share neighbors.
 * Thus a vertex in an input frontier map to multiple output items. An efficient
 * advance is the most significant challenge of a GPU implementation.
 *
 * @note Advance does not remove self-loops, i.e., a vertex can be a neighbor of
 * itself.
 *
 * @par Example
 *  The following code is a simple snippet on how to use advance within the
 * enactor loop.
 *
 *  \code
 *  // C++ lambda operator to be used within advance.
 *  auto sample = [=] __host__ __device__(
 *                          vertex_t const& source,    // ... source
 *                          vertex_t const& neighbor,  // neighbor
 *                          edge_t const& edge,        // edge
 *                          weight_t const& weight     // weight (tuple).
 *                          ) -> bool {
 *    return true; // keeps the neighbor in the output frontier.
 *  };
 *
 *  // Execute advance operator on the provided lambda.
 *  operators::advance::execute<operators::load_balance_t::block_mapped,
 *                              operators::advance_direction_t::forward,
 *                              operators::advance_io_type_t::vertex,
 *                              operators::advance_io_type_t::vertex>(
 *    G, sample, input_frontier, output_frontier, storage, context);
 *  \endcode
 *
 * @tparam lb `gunrock::operators::load_balance_t` enum, determines which
 * load-balancing algorithm to use when running advance.
 * @tparam direction `gunrock::operators::advance_direction_t` enum.
 * Determines the direction when advancing the input frontier (foward, backward,
 * both).
 * @tparam graph_t `gunrock::graph_t` struct.
 * @tparam operator_t is the type of the lambda function being passed in.
 * @tparam frontier_t `gunrock::frontier_t`.
 * @tparam work_tiles_t type of the storaged space for scanned items.
 *
 * @param G input graph used to determine the neighboring vertex/edges of the
 * input frontier.
 * @param op lambda operator to apply to each source, neighbor, edge (and
 * weight) tuple.
 * @param input input frontier being passed in.
 * @param output resultant output frontier containing the neighbors.
 * @param segments storaged space for scanned items (segment offsets).
 * @param context a `cuda::multi_context_t` that contains GPU contexts for the
 * available CUDA devices. Used to launch the advance kernels.
 */
template <load_balance_t lb,
          advance_direction_t direction,
          advance_io_type_t input_type,
          advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             gcuda::multi_context_t& context) {
  if (context.size() == 1) {
    auto context0 = context.get_context(0);

    if (lb == load_balance_t::thread_mapped) {
      thread_mapped::execute<direction, input_type, output_type>(
          G, op, *input, *output, segments, *context0);
#if __HIP_PLATFORM_NVIDIA__
    } else if (lb == load_balance_t::merge_path_v2) {
      merge_path_v2::execute<direction, input_type, output_type>(
          G, op, *input, *output, segments, *context0);
#endif
    } else if (lb == load_balance_t::merge_path) {
      merge_path::execute<direction, input_type, output_type>(
          G, op, input, output, segments, *context0);
    } else if (lb == load_balance_t::block_mapped) {
      block_mapped::execute<direction, input_type, output_type>(
          G, op, *input, *output, *context0);
    } else {
      error::throw_if_exception(hipErrorUnknown, "Advance type not supported.");
    }

  } else {
    error::throw_if_exception(hipErrorUnknown,
                              "`context.size() != 1` not supported");
  }
}

/**
 * @brief An advance operator generates a new frontier from an input frontier
 * by visiting the neighbors of the input frontier.
 *
 * @par Overview
 * A frontier can consist of either vertices or edges, and an advance step can
 * input and output either kind of frontier. The output frontier only contains
 * the neighbors and not the source vertex/edges of the input frontier. Advance
 * is an irregularly-parallel operation for two reasons:
 *  1. different vertices in a graph have different numbers of neighbors and
 *  2. vertices share neighbors.
 * Thus a vertex in an input frontier map to multiple output items. An efficient
 * advance is the most significant challenge of a GPU implementation.
 *
 * @note Advance does not remove self-loops, i.e., a vertex can be a neighbor of
 * itself.
 *
 * @par Example
 *  The following code is a simple snippet on how to use advance within the
 * enactor loop.
 *
 *  \code
 *  // C++ lambda operator to be used within advance.
 *  auto sample = [=] __host__ __device__(
 *                          vertex_t const& source,    // ... source
 *                          vertex_t const& neighbor,  // neighbor
 *                          edge_t const& edge,        // edge
 *                          weight_t const& weight     // weight (tuple).
 *                          ) -> bool {
 *    return true; // keeps the neighbor in the output frontier.
 *  };
 *
 *  // Execute advance operator on the provided lambda.
 *  operators::advance::execute<operators::load_balance_t::block_mapped,
 *                              operators::advance_direction_t::forward,
 *                              operators::advance_io_type_t::vertex,
 *                              operators::advance_io_type_t::vertex>(
 *    G, E, sample, context);
 *  \endcode
 *
 * @tparam lb `gunrock::operators::load_balance_t` enum, determines which
 * load-balancing algorithm to use when running advance.
 * @tparam direction `gunrock::operators::advance_direction_t` enum.
 * Determines the direction when advancing the input frontier (foward, backward,
 * both).
 * @tparam graph_t `gunrock::graph_t` struct.
 * @tparam enactor_type is the type of the enactor being passed in.
 * @tparam operator_type is the type of the lambda function being passed in.
 *
 * @param G input graph used to determine the neighboring vertex/edges of the
 * input frontier.
 * @param E gunrock enactor, holds the enactor pointers to input, output
 * frontier buffers and storage space for work segments.
 * @param op lambda operator to apply to each source, neighbor, edge (and
 * weight) tuple.
 * @param context a `cuda::multi_context_t` that contains GPU contexts for the
 * available CUDA devices. Used to launch the advance kernels.
 * @param swap_buffers (default = `true`), swap input and output buffers of the
 * enactor, such that the input buffer gets reused as the output buffer in the
 * next iteration. Use `false` to disable the swap behavior.
 */
template <load_balance_t lb = load_balance_t::merge_path,
          advance_direction_t direction = advance_direction_t::forward,
          advance_io_type_t input_type = advance_io_type_t::vertices,
          advance_io_type_t output_type = advance_io_type_t::vertices,
          typename graph_t,
          typename enactor_type,
          typename operator_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             gcuda::multi_context_t& context,
             bool swap_buffers = true) {
  execute<lb, direction, input_type, output_type>(
      G,                         // graph
      op,                        // advance operator
      E->get_input_frontier(),   // input frontier
      E->get_output_frontier(),  // output frontier
      E->scanned_work_domain,    // work segments
      context                    // gpu context
  );

  /*!
   * @note if the Enactor interface is used, we, the library writers assume
   * control of the frontiers and swap the input/output buffers as needed,
   * meaning; Swap frontier buffers, output buffer now becomes the input buffer
   * and vice-versa. This can be overridden by `swap_buffers`.
   */
  if (swap_buffers && (output_type != advance_io_type_t::none))
    E->swap_frontier_buffers();
}

/**
 * @brief Runtime dispatch version of advance execute that accepts load_balance_t
 * as a runtime parameter instead of a template parameter.
 * 
 * This allows algorithms to select the load balancing strategy at runtime based
 * on command-line arguments or configuration.
 * 
 * @tparam graph_t Graph type.
 * @tparam enactor_type Enactor type.
 * @tparam operator_type Operator type (lambda function).
 * @param G Input graph.
 * @param E Gunrock enactor.
 * @param op Lambda operator to apply.
 * @param lb Load balancing technique (runtime parameter).
 * @param context GPU context.
 * @param swap_buffers Whether to swap input/output buffers (default: true).
 */
template <typename graph_t,
          typename enactor_type,
          typename operator_type>
void execute_runtime(graph_t& G,
                     enactor_type* E,
                     operator_type op,
                     load_balance_t lb,
                     gcuda::multi_context_t& context,
                     bool swap_buffers = true) {
  // Dispatch to appropriate template instantiation based on runtime enum value
  if (lb == load_balance_t::thread_mapped) {
    execute<load_balance_t::thread_mapped, advance_direction_t::forward,
            advance_io_type_t::vertices, advance_io_type_t::vertices>(
        G, E, op, context, swap_buffers);
  } else if (lb == load_balance_t::block_mapped) {
    execute<load_balance_t::block_mapped, advance_direction_t::forward,
            advance_io_type_t::vertices, advance_io_type_t::vertices>(
        G, E, op, context, swap_buffers);
  } else if (lb == load_balance_t::merge_path) {
    execute<load_balance_t::merge_path, advance_direction_t::forward,
            advance_io_type_t::vertices, advance_io_type_t::vertices>(
        G, E, op, context, swap_buffers);
#if __HIP_PLATFORM_NVIDIA__
  } else if (lb == load_balance_t::merge_path_v2) {
    execute<load_balance_t::merge_path_v2, advance_direction_t::forward,
            advance_io_type_t::vertices, advance_io_type_t::vertices>(
        G, E, op, context, swap_buffers);
#endif
  } else {
    error::throw_if_exception(hipErrorUnknown, "Load balance type not supported.");
  }
}

}  // namespace advance
}  // namespace operators
}  // namespace gunrock
