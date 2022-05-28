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

// #define LBS_SEGREDUCE 1

namespace gunrock {
namespace operators {
namespace neighborreduce {

/**
 * @brief Neighbor reduce is an operator that performs reduction on the segments
 * of neighbors (or data associated with the neighbors), where each segment is
 * defined by the source vertex. Another simple way to understand this operator
 * is to perform advance and then reduction on the resultant traversal.
 *
 * @par Overview
 * Neighbor reduce operator, built on top of segmented reduction. This is
 * a very limited approach to neighbor reduce, and only gives you the edge per
 * advance. It's only implemented on the entire graph (frontiers not yet
 * supported).
 *
 * @tparam input_t advance input type (advance_io_type_t::graph supported)
 * @tparam graph_t graph type.
 * @tparam enactor_t enactor type.
 * @tparam output_t output type.
 * @tparam operator_t user-defined lambda function.
 * @tparam arithmetic_t binary function, arithmetic operator such as sum, max,
 * min, etc.
 * @param G graph to perform advance-reduce on.
 * @param E enactor structure (not used as of right now).
 * @param output output buffer.
 * @param op user-defined lambda function.
 * @param arithmetic_op arithmetic operator (binary).
 * @param init_value initial value for the reduction.
 * @param context cuda context (@see gcuda::multi_context_t).
 */
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
             gcuda::multi_context_t& context) {
  if (context.size() == 1) {
    auto context0 = context.get_context(0);

    using type_t = typename graph_t::vertex_type;
    using find_csr_t = typename graph_t::graph_csr_view_t;
    if (!(G.template contains_representation<find_csr_t>())) {
      error::throw_if_exception(cudaErrorUnknown,
                                "CSR sparse-matrix representation "
                                "required for neighborreduce operator.");
    }

#ifndef LBS_SEGREDUCE
    // TODO: Throw an exception if input_t is not advance_io_type_t::graph.
    mgpu::transform_segreduce(op, G.get_number_of_edges(), G.get_row_offsets(),
                              G.get_number_of_vertices(), output, arithmetic_op,
                              init_value, *(context0->mgpu()));

#else

    auto f = [=] __device__(std::size_t index, std::size_t seg,
                            std::size_t rank) {
      auto v = type_t(seg);
      auto start_edge = G.get_starting_edge(v);
      auto e = start_edge + rank;
      return op(e);
    };

    // TODO: Throw an exception if input_t is not advance_io_type_t::graph.
    mgpu::lbs_segreduce(f, G.get_number_of_edges(), G.get_row_offsets(),
                        G.get_number_of_vertices(), output, arithmetic_op,
                        init_value, *(context0->mgpu()));
#endif
  }
}

}  // namespace neighborreduce
}  // namespace operators
}  // namespace gunrock