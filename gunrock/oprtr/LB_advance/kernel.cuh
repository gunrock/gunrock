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
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/device_intrinsics.cuh>

#include <gunrock/oprtr/1D_oprtr/1D_scalar.cuh>
#include <gunrock/oprtr/advance/advance_base.cuh>
#include <gunrock/oprtr/LB_advance/kernel_policy.cuh>
#include <gunrock/oprtr/LB_advance/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace LB {

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
                       //#ifdef __CUDA_ARCH__
                       //    __CUDA_ARCH__,                      // CUDA_ARCH
                       //#else
                       //    0,
                       //#endif
                       1,          // MAX_CTA_OCCUPANCY
                       10,         // LOG_THREADS
                       9,          // LOG_BLOCKS
                       128 * 1024  // LIGHT_EDGE_THRESHOLD (used for partitioned
                                   // advance mode)
                       >
      KernelPolicyT;

  template <typename AdvanceOpT>
  static __device__ __forceinline__ void RelaxPartitionedEdges2(
      const GraphT &graph, const InKeyT *&keys_in, const SizeT &num_inputs,
      const SizeT *&output_offsets, const SizeT *&block_input_starts,
      // const SizeT     &partition_size,
      // const SizeT     &num_partitions,
      OutKeyT *&keys_out, ValueT *&values_out, const SizeT &num_outputs,
      // const ValueT   *&reduce_values_in,
      //      ValueT   *&reduce_values_out,
      AdvanceOpT advance_op) {
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;
    SizeT outputs_per_block = (num_outputs + gridDim.x - 1) / gridDim.x;
    SizeT block_output_start = (SizeT)blockIdx.x * outputs_per_block;
    if (block_output_start >= num_outputs) return;
    SizeT block_output_end =
        min(block_output_start + outputs_per_block, num_outputs);
    SizeT block_output_size = block_output_end - block_output_start;
    SizeT block_output_processed = 0;

    SizeT block_input_end =
        (blockIdx.x + 1 == gridDim.x)
            ? num_inputs
            : min(block_input_starts[blockIdx.x + 1], num_inputs);
    if (block_input_end < num_inputs &&
        block_output_end >
            (block_input_end > 0 ? output_offsets[block_input_end - 1] : 0))
      block_input_end++;

    SizeT iter_input_start = block_input_starts[blockIdx.x];
    SizeT block_first_v_skip_count =
        (block_output_start != output_offsets[iter_input_start])
            ? block_output_start - (iter_input_start > 0
                                        ? output_offsets[iter_input_start - 1]
                                        : 0)
            : 0;

    // if (num_inputs > 1 && threadIdx.x == 0)
    //    printf("(%3d, %3d) block_input = [%llu, %llu), block_output = [%llu,
    //    %llu)\n",
    //        blockIdx.x, threadIdx.x,
    //        (unsigned long long)iter_input_start, (unsigned long
    //        long)block_input_end, (unsigned long long)block_output_start,
    //        (unsigned long long)block_output_end);
    while (block_output_processed < block_output_size &&
           iter_input_start < block_input_end) {
      SizeT iter_input_size = min((SizeT)KernelPolicyT::SCRATCH_ELEMENTS - 1,
                                  block_input_end - iter_input_start);
      SizeT iter_input_end = iter_input_start + iter_input_size;
      SizeT iter_output_end = iter_input_end < num_inputs
                                  ? output_offsets[iter_input_end]
                                  : num_outputs;
      iter_output_end = min(iter_output_end, block_output_end);
      SizeT iter_output_size =
          min(iter_output_end - block_output_start, block_output_size);
      iter_output_size -= block_output_processed;
      SizeT iter_output_end_offset = iter_output_size + block_output_processed;

      SizeT thread_input = iter_input_start + threadIdx.x;
      if (threadIdx.x < KernelPolicyT::SCRATCH_ELEMENTS) {
        if (thread_input <= block_input_end && thread_input < num_inputs) {
          InKeyT input_item =
              (keys_in == NULL) ? thread_input : keys_in[thread_input];
          smem_storage.output_offset[threadIdx.x] =
              output_offsets[thread_input] - block_output_start;
          smem_storage.input_queue[threadIdx.x] = input_item;
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
            // smem_storage.vertices [threadIdx.x] = input_item;
            if (util::isValid(input_item))
              smem_storage.row_offset[threadIdx.x] =
                  graph.GetNeighborListOffset(input_item);
            else
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
          } else  // if ((FLAG & OprtrType_E2V) != 0 ||
                  //    (FLAG & OprtrType_E2E) != 0))
          {
            if (util::isValid(input_item)) {
              VertexT v = graph.GetEdgeDest(input_item);
              smem_storage.vertices[threadIdx.x] = v;
              smem_storage.row_offset[threadIdx.x] =
                  graph.GetNeighborListOffset(v);
            } else {
              smem_storage.vertices[threadIdx.x] =
                  util::PreDefinedValues<VertexT>::MaxValue;
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
            }
          }
        }

        else {
          smem_storage.output_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;  // max_edges;
          smem_storage.vertices[threadIdx.x] =
              util::PreDefinedValues<VertexT>::MaxValue;  // max_vertices;
          smem_storage.input_queue[threadIdx.x] =
              util::PreDefinedValues<InKeyT>::MaxValue;  // max_vertices;
          smem_storage.row_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;
        }
      }
      __syncthreads();

      SizeT v_index = 0;
      VertexT v = 0;
      InKeyT input_item = 0;
      SizeT next_v_output_start_offset = 0;
      SizeT v_output_start_offset = 0;
      SizeT row_offset_v = 0;
      for (SizeT thread_output_offset = threadIdx.x + block_output_processed;
           thread_output_offset < iter_output_end_offset;
           thread_output_offset += KernelPolicyT::THREADS) {
        if (thread_output_offset >= next_v_output_start_offset) {
          v_index = util::BinarySearch<KernelPolicyT::SCRATCH_ELEMENTS>(
              thread_output_offset, smem_storage.output_offset);
          input_item = smem_storage.input_queue[v_index];
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0)
            v = input_item;
          else
            v = smem_storage.vertices[v_index];
          row_offset_v = smem_storage.row_offset[v_index];
          next_v_output_start_offset = smem_storage.output_offset[v_index];
          if (v_index > 0) {
            block_first_v_skip_count = 0;
            v_output_start_offset = smem_storage.output_offset[v_index - 1];
          } else
            v_output_start_offset = block_output_processed;
        }

        SizeT edge_id = row_offset_v + thread_output_offset +
                        block_first_v_skip_count - v_output_start_offset;
        VertexT u = graph.GetEdgeDest(edge_id);

        ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT>(
            v, u, edge_id, iter_input_start + v_index, input_item,
            block_output_start + thread_output_offset, keys_out, values_out,
            NULL, NULL,  // reduce_values_in, reduce_values_out,
            advance_op);

      }  // end of for thread_output_offset
      block_output_processed += iter_output_size;
      iter_input_start += iter_input_size;
      block_first_v_skip_count = 0;

      __syncthreads();
    }  // end of while
  }

  template <typename AdvanceOpT>
  static __device__ __forceinline__ void RelaxLightEdges(
      const GraphT &graph, const InKeyT *&keys_in, const SizeT num_inputs,
      const SizeT *&output_offsets, OutKeyT *&keys_out, ValueT *&values_out,
      const SizeT &num_outputs,
      // const ValueT   *&reduce_values_in,
      //      ValueT   *&reduce_values_out,
      AdvanceOpT advance_op) {
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;

    SizeT block_input_start =
        (SizeT)blockIdx.x * KernelPolicyT::SCRATCH_ELEMENTS;
    while (block_input_start < num_inputs) {
      InKeyT input_item = 0;
      SizeT block_input_end =
          (block_input_start + KernelPolicyT::SCRATCH_ELEMENTS >= num_inputs)
              ? (num_inputs - 1)
              : (block_input_start + KernelPolicyT::SCRATCH_ELEMENTS - 1);
      SizeT thread_input = block_input_start + threadIdx.x;
      SizeT block_output_start =
          (block_input_start >= 1) ? output_offsets[block_input_start - 1] : 0;
      SizeT block_output_end = output_offsets[block_input_end];
      SizeT block_output_size = block_output_end - block_output_start;
      if (threadIdx.x < KernelPolicyT::SCRATCH_ELEMENTS) {
        if (thread_input <= block_input_end + 1 && thread_input < num_inputs) {
          input_item = (keys_in == NULL) ? thread_input : keys_in[thread_input];
          smem_storage.input_queue[threadIdx.x] = input_item;
          smem_storage.output_offset[threadIdx.x] =
              output_offsets[thread_input] - block_output_start;
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
            // smem_storage.vertices[threadIdx.x] = input_item;
            if (util::isValid(input_item))
              smem_storage.row_offset[threadIdx.x] =
                  graph.GetNeighborListOffset(input_item);
            else
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
          } else  // if ((FLAG & OprtrType_E2V) != 0 ||
                  //    (FLAG & OprtrType_E2E) != 0))
          {
            if (util::isValid(input_item)) {
              VertexT v = graph.GetEdgeDest(input_item);
              smem_storage.vertices[threadIdx.x] = v;
              smem_storage.row_offset[threadIdx.x] =
                  graph.GetNeighborListOffset(v);
            } else {
              smem_storage.vertices[threadIdx.x] =
                  util::PreDefinedValues<VertexT>::MaxValue;
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
            }
          }
        }  // end of if thread_input < input_queue_length

        else {
          smem_storage.output_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;  // max_edges;
          smem_storage.vertices[threadIdx.x] =
              util::PreDefinedValues<VertexT>::MaxValue;  // max_vertices;
          smem_storage.input_queue[threadIdx.x] =
              util::PreDefinedValues<InKeyT>::MaxValue;  // max_vertices;
          smem_storage.row_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;
        }
      }
      __syncthreads();

      SizeT v_index = 0;
      VertexT v = 0;
      SizeT next_v_output_start_offset = 0;
      SizeT v_output_start_offset = 0;
      SizeT row_offset_v = 0;

      for (SizeT thread_output = threadIdx.x;
           thread_output - threadIdx.x < block_output_size;
           thread_output += KernelPolicyT::THREADS) {
        if (thread_output < block_output_size) {
          if (thread_output >= next_v_output_start_offset) {
            v_index = util::BinarySearch<KernelPolicyT::SCRATCH_ELEMENTS>(
                thread_output, smem_storage.output_offset);
            input_item = smem_storage.input_queue[v_index];
            if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0)
              v = input_item;
            else
              v = smem_storage.vertices[v_index];
            v_output_start_offset =
                (v_index > 0) ? smem_storage.output_offset[v_index - 1] : 0;
            next_v_output_start_offset =
                (v_index < KernelPolicyT::SCRATCH_ELEMENTS)
                    ? smem_storage.output_offset[v_index]
                    : util::PreDefinedValues<SizeT>::MaxValue;
            row_offset_v = smem_storage.row_offset[v_index];
          }

          SizeT edge_id = row_offset_v - v_output_start_offset + thread_output;
          VertexT u = graph.GetEdgeDest(edge_id);
          // util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          //    u, column_indices + edge_id);
          ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT>(
              v, u, edge_id, block_input_start + v_index, input_item,
              block_output_start + thread_output, keys_out, values_out, NULL,
              NULL,  // reduce_values_in, reduce_values_out,
              advance_op);
        }
      }  // end of for thread_output

      block_input_start += (SizeT)gridDim.x * KernelPolicyT::SCRATCH_ELEMENTS;
      __syncthreads();
    }  // end of while
  }    // end of RelaxLightEdges
};

/**
 * @brief Kernel entry for relax partitioned edge function
 * @tparam FLAG Operator flags
 * @tparam GraphT Graph type
 * @tparam InKeyT Input keys type
 * @tparam OutKeyT Output keys type
 * @param[in] queue_reset       If reset queue length
 * @param[in] queue_index       Current frontier counter index
 * @param[in] graph             Data structure containing the graph
 * @param[in] keys_in           Pointer of InKeyT to the incoming frontier
 * @param[in] num_inputs        Length of the incoming frontier
 * @param[in] output_offsets    Pointer of SizeT to output offsets of each input
 * @param[in] partition_starts  Pointer of partition start index computed by
 * sorted search
 * @param[in] partition_size    Size of workload partition that one block
 * handles
 * @param[in] num_partitions    Number of partitions in the current frontier
 * @param[out] keys_out         Pointer of OutKeyT to the outgoing frontier
 * @param[out] values_out       Pointer of ValueT to the outgoing values
 * @param[in] num_outputs       Pointer of SizeT to the Length of the outgoing
 * frontier
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when
 * KernelPolicy::INSTRUMENT is set)
 * @param[out] values_to_reduce Pointer of ValueT to the outgoing reduced
 * results
 * @param[in] reduce_frontier   Pointer of ValueT to the incoming reduce values
 */
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename AdvanceOpT>
__launch_bounds__(
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::THREADS,
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__ void RelaxPartitionedEdges2(
        const bool queue_reset, const typename GraphT::VertexT queue_index,
        const GraphT graph, const InKeyT *keys_in,
        typename GraphT::SizeT num_inputs,
        const typename GraphT::SizeT *output_offsets,
        const typename GraphT::SizeT *block_input_starts,
        // const typename GraphT::SizeT    partition_size,
        // const typename GraphT::SizeT    num_partitions,
        OutKeyT *keys_out, typename GraphT::ValueT *values_out,
        typename GraphT::SizeT *num_outputs,
        util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
        // util::KernelRuntimeStats        kernel_stats,
        // const typename GraphT::ValueT  *reduce_values_in ,
        //      typename GraphT::ValueT  *reduce_values_out,
        AdvanceOpT advance_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs,
               work_progress);

  Dispatch<FLAG, GraphT, InKeyT, OutKeyT>::RelaxPartitionedEdges2(
      graph, keys_in, num_inputs, output_offsets,
      block_input_starts,  // partition_size, //num_partitions,
      keys_out, values_out, num_outputs[0],
      // reduce_values_in, reduce_values_out,
      advance_op);
}

/**
 * @brief Kernel entry for relax partitioned edge function
 * @tparam FLAG Operator flags
 * @tparam GraphT Graph type
 * @tparam InKeyT Input keys type
 * @tparam OutKeyT Output keys type
 * @param[in] queue_reset       If reset queue length
 * @param[in] queue_index       Current frontier counter index
 * @param[in] graph             Data structure containing the graph
 * @param[in] keys_in           Pointer of InKeyT to the incoming frontier
 * @param[in] num_inputs        Length of the incoming frontier
 * @param[in] output_offsets    Pointer of SizeT to output offsets of each input
 * @param[out] keys_out         Pointer of OutKeyT to the outgoing frontier
 * @param[out] values_out       Pointer of ValueT to the outgoing values
 * @param[in] num_outputs       Pointer of SizeT to the Length of the outgoing
 * frontier
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when
 * KernelPolicy::INSTRUMENT is set)
 * @param[out] values_to_reduce Pointer of ValueT to the outgoing reduced
 * results
 * @param[in] reduce_frontier   Pointer of ValueT to the incoming reduce values
 */
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename AdvanceOpT>
__launch_bounds__(
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::THREADS,
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__ void RelaxLightEdges(
        const bool queue_reset, const typename GraphT::VertexT queue_index,
        const GraphT graph, const InKeyT *keys_in,
        typename GraphT::SizeT num_inputs,
        const typename GraphT::SizeT *output_offsets, OutKeyT *keys_out,
        typename GraphT::ValueT *values_out,
        typename GraphT::SizeT *num_outputs,
        util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
        // util::KernelRuntimeStats        kernel_stats,
        // const typename GraphT::ValueT  *reduce_values_in,
        //      typename GraphT::ValueT  *reduce_values_out,
        AdvanceOpT advance_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs,
               work_progress);

  Dispatch<FLAG, GraphT, InKeyT, OutKeyT>::RelaxLightEdges(
      graph, keys_in, num_inputs, output_offsets, keys_out, values_out,
      num_outputs[0],
      // reduce_values_in, reduce_values_out,
      advance_op);
}

template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename AdvanceOpT>
__launch_bounds__(
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::THREADS,
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__
    void RelaxEdges(const bool queue_reset,
                    const typename GraphT::VertexT queue_index,
                    const GraphT graph, const InKeyT *keys_in,
                    typename GraphT::SizeT num_inputs,
                    const typename GraphT::SizeT *output_offsets,
                    const typename GraphT::SizeT *partition_starts,
                    // const typename GraphT::SizeT    partition_size,
                    // const typename GraphT::SizeT    num_partitions,
                    OutKeyT *keys_out, typename GraphT::ValueT *values_out,
                    typename GraphT::SizeT *num_outputs,
                    util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
                    // util::KernelRuntimeStats        kernel_stats,
                    // const typename GraphT::ValueT  *reduce_values_in,
                    //      typename GraphT::ValueT  *reduce_values_out,
                    AdvanceOpT advance_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs,
               work_progress);

  if (num_outputs[0] < Dispatch<FLAG, GraphT, InKeyT, OutKeyT,
                                true>::KernelPolicyT::LIGHT_EDGE_THRESHOLD) {
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT>::RelaxLightEdges(
        graph, keys_in, num_inputs, output_offsets, keys_out, values_out,
        num_outputs[0],
        // reduce_values_in, reduce_values_out,
        advance_op);
  } else {
    Dispatch<FLAG, GraphT, InKeyT, OutKeyT>::RelaxPartitionedEdges2(
        graph, keys_in, num_inputs, output_offsets,
        partition_starts,  // partition_size, //num_partitions,
        keys_out, values_out, num_outputs[0],
        // reduce_values_in, reduce_values_out,
        advance_op);
  }
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

  if (parameters.frontier->output_length[0] <
      KernelPolicyT::LIGHT_EDGE_THRESHOLD) {
    SizeT num_blocks =
        parameters.frontier->queue_length / KernelPolicyT::SCRATCH_ELEMENTS + 1;
    // printf("using RelaxLightEdges\n");
    // util::PrintMsg("output_length = " + std::to_string(parameters.frontier ->
    // output_length[0])
    //    + ", threads = " + std::to_string(KernelPolicyT::THREADS)
    //    + ", blocks = " + std::to_string(num_blocks));
    RelaxLightEdges<FLAG, GraphT, InKeyT, OutKeyT>
        <<<num_blocks, KernelPolicyT::THREADS, 0, parameters.stream>>>(
            parameters.frontier->queue_reset, parameters.frontier->queue_index,
            graph,
            (frontier_in == NULL) ? ((InKeyT *)NULL)
                                  : frontier_in->GetPointer(util::DEVICE),
            parameters.frontier->queue_length,
            parameters.frontier->output_offsets.GetPointer(util::DEVICE),
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
  } else {
    SizeT num_blocks =
        parameters.frontier->output_length[0] / 2 / KernelPolicyT::THREADS +
        1;  // LBPOLICY::BLOCKS
    if (num_blocks > KernelPolicyT::BLOCKS) num_blocks = KernelPolicyT::BLOCKS;
    // util::PrintMsg("output_length = " + std::to_string(parameters.frontier ->
    // output_length[0])
    //    + ", threads = " + std::to_string(KernelPolicyT::THREADS)
    //    + ", blocks = " + std::to_string(num_blocks)
    //    + ", block_output_starts = " + util::to_string(parameters.frontier ->
    //    block_output_starts.GetPointer(util::DEVICE))
    //    + ", length = " + std::to_string(parameters.frontier ->
    //    block_output_starts.GetSize()));
    SizeT outputs_per_block =
        (parameters.frontier->output_length[0] + num_blocks - 1) / num_blocks;

    oprtr::SetIdx_Kernel<<<1, 256, 0, parameters.stream>>>(
        parameters.frontier->block_output_starts.GetPointer(util::DEVICE),
        outputs_per_block, num_blocks);
    mgpu::SortedSearch<mgpu::MgpuBoundsLower>(
        parameters.frontier->block_output_starts.GetPointer(util::DEVICE),
        num_blocks,
        parameters.frontier->output_offsets.GetPointer(util::DEVICE),
        parameters.frontier->queue_length,
        parameters.frontier->block_input_starts.GetPointer(util::DEVICE),
        parameters.context[0]);

    RelaxPartitionedEdges2<FLAG, GraphT, InKeyT, OutKeyT>
        <<<num_blocks, KernelPolicyT::THREADS, 0, parameters.stream>>>(
            parameters.frontier->queue_reset, parameters.frontier->queue_index,
            graph,
            (frontier_in == NULL) ? ((InKeyT *)NULL)
                                  : frontier_in->GetPointer(util::DEVICE),
            parameters.frontier->queue_length,
            parameters.frontier->output_offsets.GetPointer(util::DEVICE),
            parameters.frontier->block_input_starts.GetPointer(util::DEVICE),
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
  }
  // TODO: switch REDUCE_OP for different reduce operators
  // Do segreduction using d_scanned_edges and d_reduce_frontier
  /*if (R_TYPE != gunrock::oprtr::advance::EMPTY && d_value_to_reduce &&
  d_reduce_frontier) { switch (R_OP) { case gunrock::oprtr::advance::PLUS: {
          SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
  output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
  (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context); break;
      }
      case gunrock::oprtr::advance::MULTIPLIES: {
          SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
  output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
  (Value)1, mgpu::multiplies<typename KernelPolicy::Value>(), context); break;
      }
      case gunrock::oprtr::advance::MAXIMUM: {
          SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
  output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
  (Value)INT_MIN, mgpu::maximum<typename KernelPolicy::Value>(), context);
            break;
      }
      case gunrock::oprtr::advance::MINIMUM: {
          SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
  output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
  (Value)INT_MAX, mgpu::minimum<typename KernelPolicy::Value>(), context);
            break;
      }
      default:
          //default operator is plus
          SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
  output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
  (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context); break;
    }
  }*/

  if (frontier_out != NULL) {
    parameters.frontier->queue_index++;
  }
  return retval;
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch_Light_CSR_CSC(const GraphT &graph,
                                 const FrontierInT *frontier_in,
                                 FrontierOutT *frontier_out,
                                 ParametersT &parameters, AdvanceOpT advance_op,
                                 FilterOpT filter_op) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename Dispatch<FLAG, GraphT, InKeyT, OutKeyT, true>::KernelPolicyT
      KernelPolicyT;

  cudaError_t retval = cudaSuccess;
  // load edge-expand-partitioned kernel
  if (parameters.get_output_length) {
    GUARD_CU(ComputeOutputLength<FLAG>(graph, frontier_in, parameters));
    GUARD_CU2(cudaStreamSynchronize(parameters.stream),
              "cudaStreamSynchronize failed");
  }

  SizeT num_block =
      parameters.frontier->queue_length / KernelPolicyT::SCRATCH_ELEMENTS + 1;
  // printf("using RelaxLightEdges\n");
  RelaxLightEdges<FLAG, GraphT, InKeyT, OutKeyT>
      <<<num_block, KernelPolicyT::THREADS, 0, parameters.stream>>>(
          parameters.frontier->queue_reset, parameters.frontier->queue_index,
          graph,
          (frontier_in == NULL) ? ((InKeyT *)NULL)
                                : frontier_in->GetPointer(util::DEVICE),
          parameters.frontier->queue_length,
          parameters.frontier->output_offsets.GetPointer(util::DEVICE),
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
        "LB is not implemented for given graph representation.");
  }

  template <OprtrFlag FLAG, typename FrontierInT, typename FrontierOutT,
            typename ParametersT, typename AdvanceOpT, typename FilterOpT>
  static cudaError_t Launch_Light_Csr_Csc(const GraphT &graph,
                                          const FrontierInT *frontier_in,
                                          FrontierOutT *frontier_out,
                                          ParametersT &parameters,
                                          AdvanceOpT advance_op,
                                          FilterOpT filter_op) {
    return util::GRError(
        cudaErrorInvalidDeviceFunction,
        "LB_Light is not implemented for given graph representation.");
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

  template <OprtrFlag FLAG, typename FrontierInT, typename FrontierOutT,
            typename ParametersT, typename AdvanceOpT, typename FilterOpT>
  static cudaError_t Launch_Light_Csr_Csc(const GraphT &graph,
                                          const FrontierInT *frontier_in,
                                          FrontierOutT *frontier_out,
                                          ParametersT &parameters,
                                          AdvanceOpT advance_op,
                                          FilterOpT filter_op) {
    return Launch_Light_CSR_CSC<FLAG>(graph, frontier_in, frontier_out,
                                      parameters, advance_op, filter_op);
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
                      "LB is not implemented for given graph representation.");

  return retval;
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch_Light(const GraphT &graph, const FrontierInT *frontier_in,
                         FrontierOutT *frontier_out, ParametersT &parameters,
                         AdvanceOpT advance_op, FilterOpT filter_op) {
  cudaError_t retval = cudaSuccess;

  if (GraphT::FLAG & gunrock::graph::HAS_CSR)
    retval =
        GraphT_Switch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSR) != 0>::
            template Launch_Light_Csr_Csc<FLAG>(graph, frontier_in,
                                                frontier_out, parameters,
                                                advance_op, filter_op);

  else if (GraphT::FLAG & gunrock::graph::HAS_CSC)
    retval =
        GraphT_Switch<GraphT, (GraphT::FLAG & gunrock::graph::HAS_CSC) != 0>::
            template Launch_Light_Csr_Csc<FLAG>(graph, frontier_in,
                                                frontier_out, parameters,
                                                advance_op, filter_op);

  else
    retval =
        util::GRError(cudaErrorInvalidDeviceFunction,
                      "LB is not implemented for given graph representation.");
  return retval;
}

}  // namespace LB
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
