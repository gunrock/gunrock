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
#include <gunrock/util/scan_device.cuh>

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
  }

  if (keys_out != NULL) {
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
  }
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename ParameterT>
cudaError_t ComputeOutputLength(const GraphT graph,
                                const FrontierInT *frontier_in,
                                ParameterT &parameters) {

  typedef typename GraphT::SizeT SizeT;
  typedef typename FrontierInT::ValueT InKeyT;
  cudaError_t retval = cudaSuccess;
  if (parameters.frontier->queue_length == 0) {
    oprtr::Set_Kernel<<<1, 1, 0, parameters.stream>>>(
        parameters.frontier->output_length.GetPointer(util::DEVICE), 0, 1);
    return retval;
  }

  int block_size = 512;
  SizeT num_blocks = parameters.frontier->queue_length / block_size + 1;
  if (num_blocks > 80 * 4) num_blocks = 80 * 4;
  GetEdgeCounts<<<num_blocks, block_size, 0, parameters.stream>>>(
      graph,
      (frontier_in == NULL) ? (InKeyT *)NULL
                            : frontier_in->GetPointer(util::DEVICE),
      parameters.frontier->queue_length,
      parameters.frontier->output_offsets.GetPointer(util::DEVICE), FLAG);

  util::cubInclusiveSum(parameters.frontier->cub_temp_space,
                        /*d_in=*/parameters.frontier->output_offsets,
                        /*d_out=*/parameters.frontier->output_offsets,
                        /*num_items=*/parameters.frontier->queue_length);
  // TODO(yzhwang): Add Move() that supports memcpy between different Array1D objects.
  // alternative solution: Modify cub implementation so that it can take in a device pointer
  // that stores the reduction result.
  GUARD_CU2(
          cudaMemcpy(parameters.frontier->output_length.GetPointer(util::DEVICE),
                     parameters.frontier->output_offsets.GetPointer(util::DEVICE) + parameters.frontier->queue_length-1, sizeof(SizeT), cudaMemcpyDeviceToDevice),
          "ComputeOutputLength cudaMemcpy output_offsets total length failed");

  return retval;
}

}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
