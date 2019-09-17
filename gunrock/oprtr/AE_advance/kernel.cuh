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
 * @brief Load balanced Edge Map Kernel Entry point
 */

#pragma once
#include <gunrock/util/device_intrinsics.cuh>

#include <gunrock/oprtr/advance/advance_base.cuh>
#include <gunrock/oprtr/AE_advance/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace AE {

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

/*template <typename GraphT, gunrock::graph::GraphFlag GraphN>
struct G_Switch
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    static __device__ __host__ __forceinline__
    void GetSrcDest(const GraphT &graph, const SizeT &edge_id,
        VertexT &src, VertexT &dest)
    {
        // Undefined
    }
};

template <typename GraphT>
struct G_Switch<GraphT, HAS_CSR> // CSR
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    static __device__ __host__ __forceinline__
    void GetSrcDest(const GraphT &graph, const SizeT &edge_id,
        VertexT &src, VertexT &dest)
    {
        src = Binary_Search(graph.row_offsets, edge_id, 0, graph.nodes);
        dest = graph.GetEdgeDest(edge_id);
    }
};

template <typename GraphT>
struct G_Switch<GraphT, HAS_CSC> // CSC
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    static __device__ __host__ __forceinline__
    void GetSrcDest(const GraphT &graph, const SizeT &edge_id,
        VertexT &src, VertexT &dest)
    {
        src = graph.GetEdgeDest(edge_id);
        dest = Binary_Search(graph.column_offsets, edge_id, 0, graph.nodes);
    }
};

template <typename GraphT>
struct G_Switch<GraphT, HAS_COO> // COO
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    static __device__ __host__ __forceinline__
    void GetSrcDest(const GraphT &graph, const SizeT &edge_id,
        VertexT &src, VertexT &dest)
    {
        typename GraphT::EdgePairT pair = graph.edge_pairs[edge_id];
        src  = pair.x;
        dest = pair.y;
    }
};*/

/*
 * @brief Dispatch data structure.
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

  typedef KernelPolicy<VertexT, InKeyT, SizeT, ValueT,  // Data types
                       8,                               // MAX_CTA_OCCUPANCY
                       8,                               // LOG_THREADS
                       8,                               // LOG_BLOCKS
                       8                                // OUTPUTS_PER_THREAD
                       >
      KernelPolicyT;

  static __device__ __forceinline__ void Write_Global_Output(
      typename KernelPolicyT::SmemStorage &smem_storage,
      const OutKeyT *thread_keys_out, SizeT &thread_output_count,
      OutKeyT *&keys_out, SizeT &output_pos) {
    KernelPolicyT::BlockScanT::Scan(thread_output_count, output_pos,
                                    smem_storage.scan_space);

    if (threadIdx.x + 1 == KernelPolicyT::THREADS) {
      if (output_pos + thread_output_count != 0) {
        smem_storage.block_offset = atomicAdd(smem_storage.output_counter,
                                              output_pos + thread_output_count);
      }
    }
    __syncthreads();

    if (thread_output_count != 0) {
      output_pos += smem_storage.block_offset;
      if (keys_out != NULL)
        for (SizeT i = 0; i < thread_output_count; i++) {
          util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(
              thread_keys_out[i], keys_out + output_pos + i);
        }
      thread_output_count = 0;
    }
  }

  template <typename AdvanceOpT>
  static __device__ __forceinline__ void Advance_Edges(
      const GraphT &graph, OutKeyT *&keys_out, ValueT *&values_out,
      SizeT *output_counter,
      // const ValueT   *&reduce_values_in,
      //      ValueT   *&reduce_values_out,
      AdvanceOpT advance_op) {
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;
    OutKeyT thread_outputs[KernelPolicyT::OUTPUTS_PER_THREAD];
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    SizeT edge_id = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    SizeT thread_output_count = 0;
    if (threadIdx.x == 0) smem_storage.output_counter = output_counter;
    __syncthreads();

    while (edge_id - threadIdx.x < graph.edges) {
      bool to_process = true;
      VertexT src = util::PreDefinedValues<VertexT>::InvalidValue;
      VertexT dest = util::PreDefinedValues<VertexT>::InvalidValue;
      SizeT out_pos = edge_id;  // util::PreDefinedValues<SizeT >::InvalidValue;
      OutKeyT out_key = util::PreDefinedValues<OutKeyT>::InvalidValue;

      if (edge_id < graph.edges) {
        graph.GetEdgeSrcDest(edge_id, src, dest);
        InKeyT input_item = (InKeyT)edge_id;

        out_key = ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT,
                                  AdvanceOpT>(
            src, dest, edge_id, edge_id, input_item, out_pos, (OutKeyT *)NULL,
            (ValueT *)NULL, (ValueT *)NULL, (ValueT *)NULL, advance_op);
      } else
        to_process = false;

      if (to_process) {
        thread_outputs[thread_output_count] = out_key;
        thread_output_count++;
      }

      if (__syncthreads_or(thread_output_count ==
                           KernelPolicyT::OUTPUTS_PER_THREAD)) {
        Write_Global_Output(smem_storage, thread_outputs, thread_output_count,
                            keys_out, out_pos);
      }
      edge_id += STRIDE;
    }
    // printf("(%3d, %3d)\n", blockIdx.x, threadIdx.x);

    if (__syncthreads_or(thread_output_count != 0)) {
      SizeT out_pos = util::PreDefinedValues<SizeT>::InvalidValue;
      Write_Global_Output(smem_storage, thread_outputs, thread_output_count,
                          keys_out, out_pos);
    }
  }
};

template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename AdvanceOpT>
__launch_bounds__(
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::THREADS,
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__
    void Kernel(const bool queue_reset,
                const typename GraphT::VertexT queue_index, const GraphT graph,
                OutKeyT *keys_out, typename GraphT::ValueT *values_out,
                util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
                // const typename GraphT::ValueT  *reduce_values_in ,
                //      typename GraphT::ValueT  *reduce_values_out,
                AdvanceOpT advance_op) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::VertexT VertexT;

  SizeT num_inputs = graph.edges;
  PrepareQueue(queue_reset, queue_index, num_inputs, (SizeT *)NULL,
               work_progress, true);

  Dispatch<FLAG, GraphT, InKeyT, OutKeyT>::Advance_Edges(
      graph, keys_out, values_out,
      work_progress.template GetQueueCounter<typename GraphT::VertexT>(
          queue_index + 1),
      // reduce_values_in, reduce_values_out,
      advance_op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch(const GraphT &graph, const FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   AdvanceOpT advance_op, FilterOpT filter_op) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT
      KernelPolicyT;

  cudaError_t retval = cudaSuccess;

  SizeT num_blocks =
      (graph.edges + KernelPolicyT::THREADS - 1) / KernelPolicyT::THREADS;
  if (num_blocks > KernelPolicyT::BLOCKS) num_blocks = KernelPolicyT::BLOCKS;

  Kernel<FLAG, GraphT, InKeyT, OutKeyT>
      <<<num_blocks, KernelPolicyT::THREADS, 0, parameters.stream>>>(
          parameters.frontier->queue_reset, parameters.frontier->queue_index,
          graph,
          (frontier_out == NULL) ? ((OutKeyT *)NULL)
                                 : frontier_out->GetPointer(util::DEVICE),
          (parameters.values_out == NULL)
              ? ((ValueT *)NULL)
              : parameters.values_out->GetPointer(util::DEVICE),
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

}  // namespace AE
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
