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
 * @brief Simple Advance Kernel Entrypoint
 */

#pragma once

#include <gunrock/oprtr/simple_advance/kernel_policy.cuh>
#include <gunrock/oprtr/simple_advance/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace SimpleAdvance {

template <typename GraphT, typename InKeyT, typename OutKeyT,
          typename AdvanceOpT>
__global__ void Kernel(
    const bool queue_reset, const typename GraphT::VertexT queue_index,
    GraphT graph, const InKeyT *keys_in, typename GraphT::SizeT num_inputs,
    OutKeyT *keys_out, typename GraphT::ValueT *values_out,
    typename GraphT::SizeT *num_outputs,
    util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
    AdvanceOpT advance_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs, work_progress,
               true);

  using VertexT = typename GraphT::VertexT;
  using SizeT = typename GraphT::SizeT;
  using ValueT = typename GraphT::ValueT;
  using EdgeItT =
      typename graph::ParallelIterator<VertexT, SizeT, ValueT, graph.FLAG>;
  static constexpr VertexT InvalidVertex =
      util::PreDefinedValues<VertexT>::InvalidValue;
  static constexpr ValueT InvalidValue =
      util::PreDefinedValues<ValueT>::InvalidValue;

  static constexpr uint32_t WARP_MASK = 0xFFFFFFFF;
  static constexpr uint32_t WARP_SIZE = 32;

  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  SizeT enqueue_offset;

  if ((tid - laneId) >= num_inputs) return;

  bool to_advance = false;
  InKeyT myKey;
  if (tid < num_inputs) {
    myKey = keys_in[tid];
    to_advance = true;
  }

  uint32_t work_queue = __ballot_sync(WARP_MASK, to_advance);

  while (work_queue) {
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_vertex = __shfl_sync(WARP_MASK, myKey, src_lane, WARP_SIZE);

    VertexT neighbor_id = 0;
    ValueT neighbor_value = 0;
    EdgeItT src_vertex_it = EdgeItT(src_vertex, &graph);
    SizeT src_vertex_size = src_vertex_it.size();
    SizeT num_warp_iterations = (src_vertex_size + WARP_SIZE - 1) / WARP_SIZE;

    for (SizeT iter = 0; iter < num_warp_iterations; iter++) {
      SizeT offset = iter * WARP_SIZE + laneId;
      neighbor_id = offset < src_vertex_size ? src_vertex_it.neighbor(offset)
                                             : InvalidVertex;
      neighbor_value =
          offset < src_vertex_size ? src_vertex_it.value(offset) : InvalidValue;

      const VertexT input_item = 0;
      SizeT input_pos = 0;
      SizeT output_pos = 0;
      bool in_frontier = false;
      if (neighbor_id != InvalidVertex) {
        in_frontier = advance_op(src_vertex, neighbor_id, neighbor_value,
                                 input_item, input_pos, output_pos);
      }

      if (keys_out != NULL && in_frontier) {
        enqueue_offset = work_progress.Enqueue(1, queue_index + 1);
        util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(
            neighbor_id, keys_out + enqueue_offset);
      }
    }

    if (laneId == src_lane) to_advance = false;
    work_queue = __ballot_sync(WARP_MASK, to_advance);
  }
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch_CSR_DYN(const GraphT &graph, const FrontierInT *frontier_in,
                           FrontierOutT *frontier_out, ParametersT &parameters,
                           AdvanceOpT advance_op, FilterOpT filter_op) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;

  cudaError_t retval = cudaSuccess;

  uint32_t queue_length = parameters.frontier->queue_length;
  uint32_t block_size = 128;
  uint32_t num_blocks = (queue_length + block_size - 1) / block_size;

  if (parameters.get_output_length) {
    GUARD_CU(ComputeOutputLength<FLAG>(graph, frontier_in, parameters));
    GUARD_CU(parameters.frontier->output_length.Move(util::DEVICE, util::HOST,
                                                     1, 0, parameters.stream));
    GUARD_CU2(cudaStreamSynchronize(parameters.stream),
              "cudaStreamSynchronize failed");
  }

  Kernel<GraphT, InKeyT, OutKeyT>
      <<<num_blocks, block_size, 0, parameters.stream>>>(
          parameters.frontier->queue_reset, parameters.frontier->queue_index,
          graph,
          (frontier_in == NULL) ? ((InKeyT *)NULL)
                                : frontier_in->GetPointer(util::DEVICE),
          parameters.frontier->queue_length,
          (frontier_out == NULL) ? ((OutKeyT *)NULL)
                                 : frontier_out->GetPointer(util::DEVICE),
          (parameters.values_out == NULL)
              ? ((ValueT *)NULL)
              : parameters.values_out->GetPointer(util::DEVICE),
          parameters.frontier->output_length.GetPointer(util::DEVICE),
          parameters.frontier->work_progress, advance_op);

  if (frontier_out != NULL) {
    parameters.frontier->queue_index++;
  }

  return retval;
}

template <typename GraphT, bool VALID>
struct GraphT_Switch {
  template <OprtrFlag FLAG, typename FrontierInT, typename FrontierOutT,
            typename ParametersT, typename AdvanceOpT, typename FilterOpT>
  static cudaError_t Launch_Csr_Dyn(const GraphT &graph,
                                    const FrontierInT *frontier_in,
                                    FrontierOutT *frontier_out,
                                    ParametersT &parameters,
                                    AdvanceOpT advance_op,
                                    FilterOpT filter_op) {
    return util::GRError(
        cudaErrorInvalidDeviceFunction,
        "Simple Advance is not implemented for given graph representation.");
  }
};
template <typename GraphT>
struct GraphT_Switch<GraphT, true> {
  template <OprtrFlag FLAG, typename FrontierInT, typename FrontierOutT,
            typename ParametersT, typename AdvanceOpT, typename FilterOpT>
  static cudaError_t Launch_Csr_Dyn(const GraphT &graph,
                                    const FrontierInT *frontier_in,
                                    FrontierOutT *frontier_out,
                                    ParametersT &parameters,
                                    AdvanceOpT advance_op,
                                    FilterOpT filter_op) {
    return Launch_CSR_DYN<FLAG>(graph, frontier_in, frontier_out, parameters,
                                advance_op, filter_op);
  }
};

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch(const GraphT &graph, const FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   AdvanceOpT advance_op, FilterOpT filter_op) {
  cudaError_t retval = cudaSuccess;
  retval = GraphT_Switch < GraphT,
  (GraphT::FLAG & gunrock::graph::HAS_CSR) ||
      (GraphT::FLAG & gunrock::graph::HAS_DYN) >
          ::template Launch_Csr_Dyn<FLAG>(graph, frontier_in, frontier_out,
                                          parameters, advance_op, filter_op);
  return retval;
}

}  // namespace SimpleAdvance
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
