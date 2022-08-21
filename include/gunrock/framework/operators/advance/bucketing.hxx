/**
 * @file bucketing.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-16
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/cuda.hxx>

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {
namespace operators {
namespace advance {
namespace bucketing {

template <advance_direction_t direction,
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
             gcuda::standard_context_t& context) {}
}  // namespace bucketing

}  // namespace advance
}  // namespace operators
}  // namespace gunrock