/**
 * @file neighborreduce.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
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

// ModernGPU support has been removed
// #include <moderngpu/kernel_segreduce.hxx>

// #define LBS_SEGREDUCE 1

namespace gunrock {
namespace operators {
namespace neighborreduce {

/**
 * @brief Neighbor reduce operator that performs reduction on the segments
 * of neighbors (or data associated with the neighbors), where each segment is
 * defined by the source vertex.
 *
 * @par Overview
 * Neighbor reduce operator, built on top of segmented reduction. This is
 * a very limited approach to neighbor reduce, and only gives you the edge per
 * advance. It's only implemented on the entire graph (frontiers not yet
 * supported).
 *
 * @tparam input_t Advance input type (advance_io_type_t::graph supported).
 * @tparam graph_t Graph type.
 * @tparam enactor_t Enactor type.
 * @tparam output_t Output type.
 * @tparam operator_t User-defined lambda function type.
 * @tparam arithmetic_t Binary function type (arithmetic operator such as sum, max, min, etc.).
 * @param G Graph to perform advance-reduce on.
 * @param E Enactor structure (not used as of right now).
 * @param output Output buffer.
 * @param op User-defined lambda function.
 * @param arithmetic_op Arithmetic operator (binary).
 * @param init_value Initial value for the reduction.
 * @param context CUDA context (@see gcuda::multi_context_t).
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
      error::throw_if_exception(hipErrorUnknown,
                                "CSR sparse-matrix representation "
                                "required for neighborreduce operator.");
    }

  // ModernGPU support has been removed. This operator is no longer available.
  error::throw_if_exception(hipErrorUnknown,
                            "neighborreduce operator is no longer supported "
                            "due to ModernGPU removal.");
  }
}

}  // namespace neighborreduce
}  // namespace operators
}  // namespace gunrock