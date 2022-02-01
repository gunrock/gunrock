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
          typename operator_t,
          typename arithmetic_t>
void execute(graph_t& G,
             enactor_t* E,
             output_t* output,
             operator_t op,
             arithmetic_t arithmetic_op,
             output_t init_value,
             cuda::multi_context_t& context) {
  if (context.size() == 1) {
    auto context0 = context.get_context(0);

    using find_csr_t = typename graph_t::graph_csr_view_t;
    if (!(G.template contains_representation<find_csr_t>())) {
      error::throw_if_exception(cudaErrorUnknown,
                                "CSR sparse-matrix representation "
                                "required for neighborreduce operator.");
    }

    // TODO: Throw an exception if input_t is not advance_io_type_t::graph.
    mgpu::transform_segreduce(op, G.get_number_of_edges(), G.get_row_offsets(),
                              G.get_number_of_vertices(), output, arithmetic_op,
                              init_value, *(context0->mgpu()));
  }
}

}  // namespace neighborreduce
}  // namespace operators
}  // namespace gunrock