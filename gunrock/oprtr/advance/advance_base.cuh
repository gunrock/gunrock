// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * advance_base.cuh
 *
 * @brief common routines for advance kernels
 */

#pragma once

#include <gunrock/oprtr/1D_oprtr/1D_scalar.cuh>
#include <gunrock/oprtr/oprtr_base.cuh>
#include <gunrock/oprtr/oprtr_parameters.cuh>

namespace gunrock {
namespace oprtr {

/*
 * @brief Dispatch data structure.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <OprtrFlag FLAG, typename VertexT, typename InKeyT, typename OutKeyT,
          typename SizeT, typename ValueT, typename OpT>
__device__ __host__ __forceinline__ OutKeyT ProcessNeighbor(
    VertexT &src, VertexT &dest, const SizeT &edge_id, const SizeT input_pos,
    const InKeyT &input_item, SizeT output_pos, OutKeyT *keys_out,
    ValueT *values_out, const ValueT *reduce_values_in,
    ValueT *reduce_values_out, OpT op) {
  OutKeyT out_key = 0;
  if (op(src, dest, edge_id, input_item, input_pos, output_pos)) {
    // if (reduce_values_in != NULL)
    //{
    //    ValueT reduce_value;
    //    if ((FLAG & ReduceType_Vertex))
    //    {
    //        reduce_value = reduce_values_in[dest];
    //    } else if ((FLAG & ReduceType_Edge))
    //    {
    //        reduce_value = reduce_values_in[edge_id];
    //    } else if ((FLAG & ReduceType_Mask))
    //    {
    //        // use user-specified function to generate value to reduce
    //    }
    //    util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(
    //        reduce_value, reduce_values_out + output_pos);
    //}

    if ((FLAG & OprtrType_V2E) != 0 || (FLAG & OprtrType_E2E) != 0) {
      out_key = (OutKeyT)edge_id;
    } else
      out_key = (OutKeyT)dest;

    if ((FLAG & OprtrOption_Idempotence) != 0 &&
        (FLAG & OprtrOption_Mark_Predecessors) != 0 && values_out != NULL) {
      util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(
          (ValueT)src, values_out + output_pos);
    }
  } else {
    out_key = util::PreDefinedValues<OutKeyT>::InvalidValue;

    // if (reduce_values_in != NULL)
    //    util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(
    //        Reduce<ValueT, FLAG & ReduceOp_Mask>::Identity,
    //        reduce_values_out + output_pos);
  }

  if (keys_out != NULL) {
    // if (util::isValid(out_key))
    // printf("(%3d, %3d) src = %llu, dest = %llu, edge = %llu, out_key = %llu,
    // output_pos = %llu\n",
    //    blockIdx.x, threadIdx.x, (unsigned long long)src,
    //    (unsigned long long)dest, (unsigned long long)edge_id,
    //    (unsigned long long)out_key, (unsigned long long)output_pos);
    util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(out_key,
                                                      keys_out + output_pos);
  }

  return out_key;
}

template <typename VertexT, typename SizeT>
__device__ __forceinline__ void PrepareQueue(
    bool queue_reset, VertexT queue_index, SizeT &input_queue_length,
    SizeT *output_queue_length, util::CtaWorkProgress<SizeT> &work_progress,
    bool skip_output_length = false) {
  // Determine work decomposition
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // obtain problem size
    if (queue_reset) {
      work_progress.StoreQueueLength(input_queue_length, queue_index);
    } else {
      input_queue_length = work_progress.LoadQueueLength(queue_index);
    }

    if (!skip_output_length)
      work_progress.Enqueue(output_queue_length[0], queue_index + 1);

    // Reset our next outgoing queue counter to zero
    work_progress.StoreQueueLength(0, queue_index + 2);
    // work_progress.PrepResetSteal(queue_index + 1);
  } else if (threadIdx.x == 0) {
    if (!queue_reset)
      input_queue_length = work_progress.LoadQueueLength(queue_index);
  }

  // Barrier to protect work decomposition
  __syncthreads();
}

template <typename GraphT, typename InKeyT>
__global__ void GetEdgeCounts(const GraphT graph, const InKeyT *keys_in,
                              const typename GraphT::SizeT num_elements,
                              typename GraphT::SizeT *edge_counts,
                              const OprtrFlag flag) {
  typedef typename GraphT::SizeT SizeT;
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;

  for (SizeT i = (SizeT)blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
       i += STRIDE) {
    InKeyT v = 0;
    if ((flag & OprtrType_V2E) != 0 || (flag & OprtrType_V2V) != 0)
      v = ((keys_in == NULL) ? i : keys_in[i]);
    else
      v = graph.GetEdgeDest((keys_in == NULL) ? i : keys_in[i]);
    edge_counts[i] = graph.GetNeighborListLength(v);
    // printf("(%3d, %3d) v = %lld, edge_counts[%lld] = %lld\n",
    //    blockIdx.x, threadIdx.x, (unsigned long long)v,
    //    (unsigned long long)i, (unsigned long long)(edge_counts[i]));
  }
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename ParameterT>
cudaError_t ComputeOutputLength(const GraphT graph,
                                const FrontierInT *frontier_in,
                                ParameterT &parameters) {
  // Load Load Balanced Kernel
  // Get Rowoffsets
  // Use scan to compute edge_offsets for each vertex in the frontier
  // Use sorted sort to compute partition bound for each work-chunk
  // load edge-expand-partitioned kernel

  typedef typename GraphT::SizeT SizeT;
  typedef typename FrontierInT::ValueT InKeyT;
  cudaError_t retval = cudaSuccess;
  if (parameters.frontier->queue_length == 0) {
    // printf("setting output_length to 0");
    oprtr::Set_Kernel<<<1, 1, 0, parameters.stream>>>(
        parameters.frontier->output_length.GetPointer(util::DEVICE), 0, 1);
    return retval;
  }

  // util::PrintMsg("output_offsets.size() = "
  //    + std::to_string(parameters.frontier -> output_offsets.GetSize())
  //    + ", queue_length = "
  //    + std::to_string(parameters.frontier -> queue_length)
  //    + ", queue_size = "
  //    + std::to_string(frontier_in == NULL ? 0 : frontier_in -> GetSize()));

  int block_size = 512;
  SizeT num_blocks = parameters.frontier->queue_length / block_size + 1;
  if (num_blocks > 80 * 4) num_blocks = 80 * 4;
  GetEdgeCounts<<<num_blocks, block_size, 0, parameters.stream>>>(
      graph,
      (frontier_in == NULL) ? (InKeyT *)NULL
                            : frontier_in->GetPointer(util::DEVICE),
      parameters.frontier->queue_length,  // TODO: +1?
      parameters.frontier->output_offsets.GetPointer(util::DEVICE), FLAG);
  // util::DisplayDeviceResults(partitioned_scanned_edges,
  // frontier_attribute->queue_length);

  mgpu::Scan<mgpu::MgpuScanTypeInc>(
      parameters.frontier->output_offsets.GetPointer(util::DEVICE),
      parameters.frontier->queue_length,  // TODO: +1?
      (SizeT)0, mgpu::plus<SizeT>(),
      parameters.frontier->output_length.GetPointer(util::DEVICE),
      (SizeT *)NULL,
      parameters.frontier->output_offsets.GetPointer(util::DEVICE),
      parameters.context[0]);

  return retval;
}

}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
