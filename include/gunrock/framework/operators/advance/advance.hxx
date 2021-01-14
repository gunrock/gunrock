/**
 * @file advance.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <bits/stdc++.h>
#include <gunrock/cuda/context.hxx>
#include <gunrock/error.hxx>
#include <gunrock/util/type_limits.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <gunrock/framework/operators/advance/helpers.hxx>
#include <gunrock/framework/operators/advance/merge_path.hxx>
#include <gunrock/framework/operators/advance/input_oriented.hxx>
#include <gunrock/framework/operators/advance/all_edges.hxx>

namespace gunrock {
namespace operators {
namespace advance {

/**
 * @brief
 *
 * @tparam type
 * @tparam direction
 * @tparam lb
 * @tparam graph_t
 * @tparam operator_t
 * @tparam frontier_t
 * @tparam work_tiles_t
 * @param G
 * @param op
 * @param input
 * @param output
 * @param segments
 * @param context
 */
template <advance_type_t type,
          advance_direction_t direction,
          load_balance_t lb,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             cuda::standard_context_t* context) {
  if (lb == load_balance_t::merge_path)
    merge_path::execute<type, direction>(G, op, input, output, segments,
                                         *context);
  else if (lb == load_balance_t::input_oriented)
    input_oriented::execute<type, direction>(G, op, input, output, segments,
                                             *context);
  else if (lb == load_balance_t::all_edges)
    all_edges::execute<type, direction>(G, op, input, output, segments,
                                        *context);
  else
    error::throw_if_exception(cudaErrorUnknown,
                              "Unsupported advance's load-balancing schedule.");
}

/**
 * @brief
 *
 * @tparam type
 * @tparam direction
 * @tparam lb
 * @tparam graph_t
 * @tparam enactor_type
 * @tparam operator_type
 * @param G
 * @param E
 * @param op
 * @param context
 */
template <advance_type_t type = advance_type_t::vertex_to_vertex,
          advance_direction_t direction = advance_direction_t::forward,
          load_balance_t lb = load_balance_t::merge_path,
          typename graph_t,
          typename enactor_type,
          typename operator_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t* context) {
  execute<type, direction, lb>(G,                         // graph
                               op,                        // advance operator
                               E->get_input_frontier(),   // input frontier
                               E->get_output_frontier(),  // output frontier
                               E->scanned_work_domain,    // work segments
                               context                    // gpu context
  );

  // Important note: if the Enactor interface is used, we, the library writers
  // assume control of the frontiers and swap the input/output buffers as
  // needed, meaning; Swap frontier buffers, output buffer now becomes the input
  // buffer and vice-versa.
  E->swap_frontier_buffers();
}

}  // namespace advance
}  // namespace operators
}  // namespace gunrock