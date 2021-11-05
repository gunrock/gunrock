/**
 * @file neighborreduce.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-11-05
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/cuda/context.hxx>
#include <gunrock/error.hxx>
#include <gunrock/util/type_limits.hxx>

#include <gunrock/framework/operators/configs.hxx>

#include <moderngpu/kernel_segreduce.hxx>

namespace gunrock {
namespace operators {
namespace neighborreduce {

template <advance_io_type_t input_t = advance_io_type_t::graph,
          typename graph_t,
          typename enactor_t,
          typename output_t,
          typename operator_t>
void execute(graph_t& G,
             enactor_t* E,
             output_t* output,
             operator_t op,
             cuda::multi_context_t& context) {
  if (context.size() == 1) {
    auto context0 = context.get_context(0);

    // TODO: Expose the binary op and init value.
    // TODO: Throw an exception if CSR isn't supported.
    // TODO: Throw an exception if input_t is not advance_io_type_t::graph.
    mgpu::transform_segreduce(op, G.get_number_of_edges(), G.get_row_offsets(),
                              G.get_number_of_vertices(), output,
                              mgpu::plus_t<output_t>(), (output_t)0,
                              *(context0->mgpu()));
  }
}

}  // namespace neighborreduce
}  // namespace operators
}  // namespace gunrock