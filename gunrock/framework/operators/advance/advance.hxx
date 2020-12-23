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

#include <gunrock/framework/operators/advance/merge_path.hxx>
#include <gunrock/framework/operators/advance/unbalanced.hxx>

namespace gunrock {
namespace operators {
namespace advance {

template <advance_type_t type,
          advance_direction_t direction,
          load_balance_t lb,
          typename graph_t,
          typename enactor_type,
          typename operator_type,
          typename frontier_type>
void execute(graph_t& G,
             enactor_type* E,
             operator_type op,
             frontier_type* input,
             frontier_type* output,
             cuda::standard_context_t* context) {
  if (lb == load_balance_t::merge_path)
    merge_path::execute<type, direction>(G, E, op, input, output, *context);
  else if (lb == load_balance_t::unbalanced)
    unbalanced::execute<type, direction>(G, E, op, input, output, *context);
  else
    error::throw_if_exception(cudaErrorUnknown,
                              "Unsupported advance's load-balancing schedule.");
}

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
  execute<type, direction, lb>(G, E, op, E->get_input_frontier(),
                               E->get_output_frontier(), context);
}

}  // namespace advance
}  // namespace operators
}  // namespace gunrock