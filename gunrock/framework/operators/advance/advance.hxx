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

#include <gunrock/framework/operators/configs.hxx>

#include <gunrock/framework/operators/advance/merge_path.hxx>
#include <gunrock/framework/operators/advance/unbalanced.hxx>

namespace gunrock {
namespace operators {
namespace advance {

template <advance_type_t type = advance_type_t::vertex_to_vertex,
          advance_direction_t direction = advance_direction_t::forward,
          load_balance_t lb = load_balance_t::merge_path,
          typename graph_type,
          typename enactor_type,
          typename operator_type>
void execute(graph_type* G,
             enactor_type* E,
             operator_type op,
             cuda::standard_context_t* context) {
  if (lb == load_balance_t::merge_path)
    merge_path::execute<type, direction>(G, E, op, *context);
  else if (lb == load_balance_t::unbalanced)
    unbalanced::execute<type, direction>(G, E, op, *context);
  else
    error::throw_if_exception(cudaErrorUnknown);
}

}  // namespace advance
}  // namespace operators
}  // namespace gunrock