// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief API interfaces for neighborreduce calls
 */

#pragma once

#include <gunrock/oprtr/advance/advance_base.cuh>
#include <gunrock/util/reduce_device.cuh>

namespace gunrock {
namespace oprtr {

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
extern cudaError_t Advance(const GraphT &graph, FrontierInT *frontier_in,
                           FrontierOutT *frontier_out, ParametersT &parameters,
                           OpT op);

namespace neighborreduce {

template <typename GraphT, graph::GraphFlag FLAG>
struct GraphTypeSwitch {};

template <typename GraphT>
struct GraphTypeSwitch<GraphT, graph::HAS_CSR> {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  static const util::Array1D<SizeT, SizeT> &GetOffsets(const GraphT &graph) {
    return graph.CsrT::row_offsets;
  }
};

template <typename GraphT>
struct GraphTypeSwitch<GraphT, graph::HAS_CSC> {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CscT CscT;

  static const util::Array1D<SizeT, SizeT> &GetOffsets(const GraphT &graph) {
    return graph.CscT::column_offsets;
  }
};

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOp,
          typename ReduceOp>
cudaError_t Launch(const GraphT &graph, FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   AdvanceOp advance_op, ReduceOp reduce_op,
                   typename ParametersT::ValueT init_value) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  cudaError_t retval = cudaSuccess;

  if ((parameters.advance_mode == "LB" ||
       parameters.advance_mode == "LB_LIGHT" ||
       (parameters.advance_mode == "ALL_EDGES" &&
        ((GraphT::FLAG & graph::HAS_CSR) != 0 ||
         (GraphT::FLAG & graph::HAS_CSC) != 0))) &&
      ((FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_SRC ||
       (FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_INPUT_POS)) {
    auto &frontier = parameters.frontier[0];
    auto &values_temp = parameters.reduce_values_temp[0];
    // GUARD_CU(values_temp.EnsureSize_(
    //    parameters.frontier.queue_length, util::DEVICE));

    // util::PrintMsg("Advance Start, values_temp = "
    //    + util::to_string(values_temp.GetPointer(util::DEVICE))
    //    + ", size = " + std::to_string(values_temp.GetSize()));
    GUARD_CU(oprtr::Advance<FLAG>(
        graph, frontier_in, frontier_out, parameters,
        [advance_op, values_temp, init_value] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
          bool retval = true;
          ValueT val =
              advance_op(src, dest, edge_id, input_item, input_pos, output_pos);
          if (!util::isValid(val)) {
            val = init_value;
            retval = false;
          }
          // if (output_pos < 0 || output_pos >= values_temp.GetSize())
          //    printf("Invalid : output_pos = %d, input_pos = %d, "
          //        "edge %d, %d -> %d\n",
          //        output_pos, input_pos, edge_id, src, dest);
          // printf("values_temp[%d] : %f <- %f\n",
          //    output_pos, values_temp[output_pos], val);
          values_temp[output_pos] = val;
          return retval;
        }));

    auto values_temp2 = parameters.reduce_values_temp2;
    if ((FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_INPUT_POS)
      values_temp2 = parameters.reduce_values_out;
    auto &values_temp2_ = values_temp2[0];
    auto offsets = frontier.segment_offsets;
    // GUARD_CU(cudaStreamSynchronize(parameters.stream));
    // util::PrintMsg("Past Advance");

    // GUARD_CU(values_temp2_.ForEach(
    //    [] __host__ __device__ (ValueT &val)
    //    {
    //        val = 0;
    //    }, parameters.advance_mode == "ALL_EDGES" ? graph.nodes :
    //    frontier.queue_length, util::DEVICE, parameters.stream));

    typedef GraphTypeSwitch<GraphT,
                            GraphT::FLAG &(graph::HAS_CSR | graph::HAS_CSC)>
        GraphSwitchT;
    if (parameters.advance_mode == "ALL_EDGES") {
      offsets = const_cast<typeof(offsets)>(&(GraphSwitchT::GetOffsets(graph)));
      GUARD_CU(util::SegmentedReduce(frontier.cub_temp_space, values_temp,
                                     values_temp2_, (SizeT)graph.nodes,
                                     offsets[0], reduce_op, init_value,
                                     parameters.stream));
    } else {
      GUARD_CU(util::SegmentedReduce(frontier.cub_temp_space, values_temp,
                                     values_temp2_, frontier.queue_length,
                                     frontier.segment_offsets[0], reduce_op,
                                     init_value, parameters.stream));
    }
    // GUARD_CU(cudaStreamSynchronize(parameters.stream));
    // util::PrintMsg("Past SegReduce");

    bool reduce_reset = parameters.reduce_reset;
    if ((FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_SRC &&
        frontier_in != NULL) {
      auto &keys_in = frontier_in[0];
      GUARD_CU(parameters.reduce_values_out->ForAll(
          [values_temp2_, keys_in, reduce_reset, reduce_op] __host__ __device__(
              ValueT * vals, const SizeT &pos) {
            SizeT val_pos = keys_in[pos];
            vals[val_pos] = reduce_reset
                                ? values_temp2_[pos]
                                : reduce_op(vals[val_pos], values_temp2_[pos]);
          },
          frontier.queue_length, util::DEVICE, parameters.stream));
    } else {
      GUARD_CU(parameters.reduce_values_out->ForAll(
          [values_temp2_, reduce_reset, reduce_op] __host__ __device__(
              ValueT * vals, const SizeT &pos) {
            ValueT new_val = reduce_reset
                                 ? values_temp2_[pos]
                                 : reduce_op(vals[pos], values_temp2_[pos]);
            // printf("vals[%d] : %f <- %f\n",
            //    pos, vals[pos], new_val);
            vals[pos] = new_val;
          },
          parameters.advance_mode == "ALL_EDGES" ? graph.nodes
                                                 : frontier.queue_length,
          util::DEVICE, parameters.stream));
    }
    // GUARD_CU(cudaStreamSynchronize(parameters.stream));
    // util::PrintMsg("Past Val Assigment");
  }

  else if (parameters.advance_mode == "TWC" &&
           ((FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_SRC ||
            (FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_INPUT_POS)) {
    GUARD_CU2(
        cudaErrorNotSupported,
        "NeighborReduce with TWC advance and reduce to source or input_pos "
        "has not been implemented yet.");
  }

  else if ((FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_DEST ||
           (parameters.advance_mode == "ALL_EDGES" &&
            (GraphT::FLAG & graph::HAS_COO) != 0)) {
    bool to_dest = ((FLAG & OprtrMode_ReduceMask) == OprtrMode_REDUCE_TO_DEST);
    auto &values_out = parameters.reduce_values_out[0];
    if (parameters.reduce_reset) {
      GUARD_CU(values_out.ForEach(
          [init_value] __host__ __device__(ValueT & val) { val = init_value; },
          graph.nodes, util::DEVICE, parameters.stream));
    }

    GUARD_CU(oprtr::Advance<FLAG>(
        graph, frontier_in, frontier_out, parameters,
        [advance_op, values_out, to_dest, reduce_op] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
          ValueT val =
              advance_op(src, dest, edge_id, input_item, input_pos, output_pos);
          if (!util::isValid(val)) return false;

          SizeT val_pos = src;
          if (to_dest) val_pos = dest;
          ValueT old_val = values_out[val_pos];
          ValueT expected;
          do {
            expected = old_val;
            old_val = atomicCAS(values_out + val_pos, expected,
                                reduce_op(old_val, val));
          } while (expected != old_val);
          return true;
        }));
  }

  else {
    GUARD_CU2(cudaErrorNotSupported,
              "Advance mode " + parameters.advance_mode +
                  " and specific graph type +"
                  " reduce target combination is not supported");
  }

  return retval;
}

}  // namespace neighborreduce
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
