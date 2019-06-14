// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * oprtr.cuh
 *
 * @brief calling interfaces of operators
 */

#pragma once

#include <gunrock/oprtr/oprtr_base.cuh>
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/intersection/kernel.cuh>
#include <gunrock/oprtr/neighborreduce/kernel.cuh>
#include <gunrock/oprtr/1D_oprtr/for.cuh>

namespace gunrock {
namespace oprtr {

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
cudaError_t Advance(const GraphT &graph, FrontierInT *frontier_in,
                    FrontierOutT *frontier_out, ParametersT &parameters,
                    OpT op) {
  return oprtr::advance::Launch<FLAG>(graph, frontier_in, frontier_out,
                                      parameters, op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Advance(const GraphT &graph, FrontierInT *frontier_in,
                    FrontierOutT *frontier_out, ParametersT &parameters,
                    AdvanceOpT advance_op, FilterOpT filter_op) {
  return oprtr::advance::Launch<FLAG>(graph, frontier_in, frontier_out,
                                      parameters, advance_op, filter_op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
cudaError_t Filter(const GraphT graph, FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   OpT op) {
  return oprtr::filter::Launch<FLAG>(graph, frontier_in, frontier_out,
                                     parameters, op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename Op1T,
          typename Op2T>
cudaError_t Filter(const GraphT graph, FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   Op1T op1, Op2T op2) {
  return oprtr::filter::Launch<FLAG>(graph, frontier_in, frontier_out,
                                     parameters, op1, op2);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
cudaError_t Compute(const GraphT graph, FrontierInT *frontier_in,
                    FrontierOutT *frontier_out, ParametersT &parameters,
                    OpT op) {
  // return oprtr::compute::Launch<FLAG>(
  //    graph, frontier_in, frontier_out, parameters, op);
  return util::GRError(cudaErrorInvalidValue, "Compute Kernel undefined.",
                       __FILE__, __LINE__);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename Op1T,
          typename Op2T>
cudaError_t Compute(const GraphT graph, FrontierInT *frontier_in,
                    FrontierOutT *frontier_out, ParametersT &parameters,
                    Op1T op1, Op2T op2) {
  // return oprtr::compute::Launch<FLAG>(
  //    graph, frontier_in, frontier_out, parameters, op1, op2);
  return util::GRError(cudaErrorInvalidValue, "Compute Kernel undefined.",
                       __FILE__, __LINE__);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOp,
          typename ReduceOp, typename ValueT>
cudaError_t NeighborReduce(const GraphT &graph, FrontierInT *frontier_in,
                           FrontierOutT *frontier_out, ParametersT &parameters,
                           AdvanceOp advance_op, ReduceOp reduce_op,
                           ValueT init_val) {
  return oprtr::neighborreduce::Launch<FLAG>(graph, frontier_in, frontier_out,
                                             parameters, advance_op, reduce_op,
                                             init_val);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOp>
cudaError_t NeighborReduce(const GraphT &graph, FrontierInT *frontier_in,
                           FrontierOutT *frontier_out, ParametersT &parameters,
                           AdvanceOp advance_op) {
  typedef typename GraphT::ValueT ValueT;
  cudaError_t retval = cudaSuccess;

  GUARD_CU(oprtr::neighborreduce::Launch<FLAG>(
      graph, frontier_in, frontier_out, parameters, advance_op,
      Reduce<ValueT, FLAG & ReduceOp_Mask>::op,
      Reduce<ValueT, FLAG & ReduceOp_Mask>::Identity));

  return retval;
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
cudaError_t Intersect(const GraphT graph, FrontierInT *frontier_in,
                      FrontierOutT *frontier_out, ParametersT &parameters,
                      OpT op) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(oprtr::intersection::Launch<FLAG>(graph, frontier_in, frontier_out,
                                             parameters, op));

  return retval;
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
cudaError_t Launch(const GraphT graph, FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   OpT op) {
  // cudaError_t retval = cudaSuccess;
  if (parameters.oprtr_type == "Advance")
    return oprtr::advance::Launch<FLAG>(graph, frontier_in, frontier_out,
                                        parameters, op);
  if (parameters.oprtr_type == "Filter")
    return oprtr::filter ::Launch<FLAG>(graph, frontier_in, frontier_out,
                                        parameters, op);
  if (parameters.oprtr_type == "Intersect")
    return oprtr::intersection ::Launch<FLAG>(graph, frontier_in, frontier_out,
                                              parameters, op);
  // if (parameters.oprtr_type == "Compute")
  //    return oprtr::compute::Launch<FLAG>(
  //        graph, frontier_in, frontier_out, parameters, op);
  return util::GRError(cudaErrorInvalidValue,
                       "OprtrType " + parameters.oprtr_type + " undefined.",
                       __FILE__, __LINE__);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename Op1T,
          typename Op2T>
cudaError_t Launch(const GraphT graph, FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   Op1T op1, Op2T op2) {
  // cudaError_t retval = cudaSuccess;
  if (parameters.oprtr_type == "Advance")
    return oprtr::advance::Launch<FLAG>(graph, frontier_in, frontier_out,
                                        parameters, op1, op2);
  if (parameters.oprtr_type == "Filter")
    return oprtr::filter ::Launch<FLAG>(graph, frontier_in, frontier_out,
                                        parameters, op1, op2);
  // if (parameters.oprtr_type == "Compute")
  //    return oprtr::compute::Launch<FLAG>(
  //        graph, frontier_in, frontier_out, parameters, op1, op2);
  return util::GRError(cudaErrorInvalidValue,
                       "OprtrType " + parameters.oprtr_type + " undefined.",
                       __FILE__, __LINE__);
}

}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
