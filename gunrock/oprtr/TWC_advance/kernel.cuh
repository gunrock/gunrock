// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

// Add Functor into Kernel Call (done)

/**
 * @file
 * kernel.cuh
 *
 * @brief Forward Edge Map Kernel Entrypoint
 */

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/TWC_advance/kernel_policy.cuh>
#include <gunrock/oprtr/TWC_advance/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace TWC {

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 * @tparam FLAG Operator flags
 * @tparam GraphT Graph type
 * @tparam InKeyT Input keys type
 * @tparam OutKeyT Output keys type
 * @tparam VALID
 */
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          bool VALID =
#ifndef __CUDA_ARCH__
              false
#else
              (__CUDA_ARCH__ >= CUDA_ARCH)
#endif
          >
struct Dispatch {
};

/**
 * @brief Kernel dispatch code for different architectures.
 * @tparam FLAG Operator flags
 * @tparam GraphT Graph type
 * @tparam InKeyT Input keys type
 * @tparam OutKeyT Output keys type
 */
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT>
struct Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true> {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;

  typedef KernelPolicy<FLAG, VertexT, InKeyT, OutKeyT, SizeT,
                       ValueT,  // Data types
                       8,       // int _MIN_CTA_OCCUPANCY, // Tunable parameters
                       7,       // int _LOG_THREADS,
                       1,       // int _LOG_LOAD_VEC_SIZE,
                       1,       // int _LOG_LOADS_PER_TILE,
                       5,       // int _LOG_RAKING_THREADS,
                       32,      // int _WARP_GATHER_THRESHOLD,
                       128 * 4,  // int _CTA_GATHER_THRESHOLD>
                       7>        // int _LOG_SCHEDULE_GRANULARITY
      KernelPolicyT;
  typedef Cta<GraphT, KernelPolicyT> CtaT;

  template <typename AdvanceOpT>
  static __device__ __forceinline__ void Kernel(
      const GraphT &graph, const InKeyT *&keys_in, const SizeT &num_inputs,
      const VertexT &queue_index,
      // const SizeT    *&output_offsets,
      // const SizeT    *&block_input_starts,
      // const SizeT     &partition_size,
      // const SizeT     &num_partitions,
      OutKeyT *&keys_out, ValueT *&values_out, const SizeT &num_outputs,
      util::CtaWorkProgress<SizeT> &work_progress,
      // const ValueT   *&reduce_values_in,
      //      ValueT   *&reduce_values_out,
      AdvanceOpT advance_op) {
    // Shared storage for the kernel
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;

    // Determine work decomposition
    if (threadIdx.x == 0) {
      // Initialize work decomposition in smem
      smem_storage.state.work_decomposition
          .template Init<KernelPolicyT::LOG_SCHEDULE_GRANULARITY>(num_inputs,
                                                                  gridDim.x);
    }

    // Barrier to protect work decomposition
    __syncthreads();

    // Determine threadblock's work range
    util::CtaWorkLimits<SizeT> work_limits;
    smem_storage.state.work_decomposition
        .template GetCtaWorkLimits<KernelPolicyT::LOG_TILE_ELEMENTS,
                                   KernelPolicyT::LOG_SCHEDULE_GRANULARITY>(
            work_limits);

    // Return if we have no work to do
    if (!work_limits.elements) {
      return;
    }

    // CTA processing abstraction
    // Cta cta(
    //  queue_reset,
    //  queue_index,
    //  label,
    //  d_row_offsets,
    //  d_inverse_row_offsets,
    //  d_column_indices,
    //  d_inverse_column_indices,
    //  d_keys_in,
    //  d_keys_out,
    //  d_values_out,
    //  d_data_slice,
    //  input_queue_length,
    //  max_in_frontier,
    //  max_out_frontier,
    //  work_progress,
    //  smem_storage,
    //  //ADVANCE_TYPE,
    //  input_inverse_graph,
    //  //R_TYPE,
    //  //R_OP,
    //  d_value_to_reduce,
    //  d_reduce_frontier);
    CtaT cta(graph, keys_in, num_inputs, keys_out, values_out, queue_index,
             // reduce_values_in, reduce_values_out,
             work_progress, smem_storage);

    // Process full tiles
    while (work_limits.offset < work_limits.guarded_offset) {
      cta.ProcessTile(work_limits.offset, KernelPolicyT::TILE_ELEMENTS,
                      advance_op);
      work_limits.offset += KernelPolicyT::TILE_ELEMENTS;
    }

    // Clean up last partial tile with guarded-i/o
    if (work_limits.guarded_elements) {
      cta.ProcessTile(work_limits.offset, work_limits.guarded_elements,
                      advance_op);
    }
  }
};

/**
 * @brief Forward edge map kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for forward edge mapping.
 * @tparam ProblemData Problem data type for forward edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset       If reset queue counter
 * @param[in] queue_index       Current frontier queue counter index
 * @param[in] label             Distance from source (label) of current frontier
 * @param[in] num_elements      Number of elements
 * @param[in] d_in_queue        Device pointer of VertexId to the incoming
 * frontier queue
 * @param[in] d_pred_out         Device pointer of VertexId to the outgoing
 * predecessor queue (only used when both mark_pred and enable_idempotence are
 * set)
 * @param[in] d_out_queue       Device pointer of VertexId to the outgoing
 * frontier queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices
 * queue
 * @param[in] problem           Device pointer to the problem object
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] max_in_queue      Maximum number of elements we can place into the
 * incoming frontier
 * @param[in] max_out_queue     Maximum number of elements we can place into the
 * outgoing frontier
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when
 * KernelPolicy::INSTRUMENT is set)
 */
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename AdvanceOpT>
__launch_bounds__(
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::THREADS,
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__
    void Kernel(const bool queue_reset,
                const typename GraphT::VertexT queue_index, const GraphT graph,
                const InKeyT *keys_in, typename GraphT::SizeT num_inputs,
                // const typename GraphT::SizeT   *output_offsets,
                // const typename GraphT::SizeT   *block_input_starts,
                // const typename GraphT::SizeT    partition_size,
                // const typename GraphT::SizeT    num_partitions,
                OutKeyT *keys_out, typename GraphT::ValueT *values_out,
                typename GraphT::SizeT *num_outputs,
                util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
                // util::KernelRuntimeStats        kernel_stats,
                // const typename GraphT::ValueT  *reduce_values_in ,
                //      typename GraphT::ValueT  *reduce_values_out,
                AdvanceOpT advance_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs, work_progress,
               true);

  Dispatch<FLAG, GraphT, InKeyT, OutKeyT>::Kernel(
      graph, keys_in, num_inputs, queue_index,  // output_offsets,
      // block_input_starts, //partition_size, //num_partitions,
      keys_out, values_out, num_outputs[0], work_progress,
      // reduce_values_in, reduce_values_out,
      advance_op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch_CSR_CSC(const GraphT &graph, const FrontierInT *frontier_in,
                           FrontierOutT *frontier_out, ParametersT &parameters,
                           AdvanceOpT advance_op, FilterOpT filter_op) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT
      KernelPolicyT;

  cudaError_t retval = cudaSuccess;
  // load edge-expand-partitioned kernel
  if (parameters.get_output_length) {
    // util::PrintMsg("getting output length");
    GUARD_CU(ComputeOutputLength<FLAG>(graph, frontier_in, parameters));
    GUARD_CU(parameters.frontier->output_length.Move(util::DEVICE, util::HOST,
                                                     1, 0, parameters.stream));
    GUARD_CU2(cudaStreamSynchronize(parameters.stream),
              "cudaStreamSynchronize failed");
    // GUARD_CU2(cudaDeviceSynchronize(),
    //    "cudaDeviceSynchronize failed");
  }

  // int blocks = 1; // TODO: calculate number of blocks
  int num_blocks =
      (parameters.max_grid_size <= 0)
          ? parameters.cuda_props->device_props.multiProcessorCount *
                KernelPolicyT::CTA_OCCUPANCY
          : parameters.max_grid_size;
  Kernel<FLAG, GraphT, InKeyT, OutKeyT>
      <<<num_blocks, KernelPolicyT::THREADS, 0, parameters.stream>>>(
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
          parameters.frontier->work_progress,
          //(parameters.reduce_values_in  == NULL) ? ((ValueT*)NULL)
          //    : (parameters.reduce_values_in  -> GetPointer(util::DEVICE)),
          //(parameters.reduce_values_out == NULL) ? ((ValueT*)NULL)
          //    : (parameters.reduce_values_out -> GetPointer(util::DEVICE)),
          advance_op);

  if (frontier_out != NULL) {
    parameters.frontier->queue_index++;
  }
  return retval;
}

template <typename GraphT, bool VALID>
struct GraphT_Switch {
  template <OprtrFlag FLAG, typename FrontierInT, typename FrontierOutT,
            typename ParametersT, typename AdvanceOpT, typename FilterOpT>
  static cudaError_t Launch_Csr_Csc(const GraphT &graph,
                                    const FrontierInT *frontier_in,
                                    FrontierOutT *frontier_out,
                                    ParametersT &parameters,
                                    AdvanceOpT advance_op,
                                    FilterOpT filter_op) {
    return util::GRError(
        cudaErrorInvalidDeviceFunction,
        "TWC is not implemented for given graph representation.");
  }
};

template <typename GraphT>
struct GraphT_Switch<GraphT, true> {
  template <OprtrFlag FLAG, typename FrontierInT, typename FrontierOutT,
            typename ParametersT, typename AdvanceOpT, typename FilterOpT>
  static cudaError_t Launch_Csr_Csc(const GraphT &graph,
                                    const FrontierInT *frontier_in,
                                    FrontierOutT *frontier_out,
                                    ParametersT &parameters,
                                    AdvanceOpT advance_op,
                                    FilterOpT filter_op) {
    return Launch_CSR_CSC<FLAG>(graph, frontier_in, frontier_out, parameters,
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

  if (GraphT::FLAG & gunrock::graph::HAS_CSR)
    retval =
        GraphT_Switch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSR) != 0>::
            template Launch_Csr_Csc<FLAG>(graph, frontier_in, frontier_out,
                                          parameters, advance_op, filter_op);

  else if (GraphT::FLAG & gunrock::graph::HAS_CSC)
    retval =
        GraphT_Switch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSC) != 0>::
            template Launch_Csr_Csc<FLAG>(graph, frontier_in, frontier_out,
                                          parameters, advance_op, filter_op);

  else
    retval =
        util::GRError(cudaErrorInvalidDeviceFunction,
                      "TWC is not implemented for given graph representation.");
  return retval;
}

}  // namespace TWC
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
