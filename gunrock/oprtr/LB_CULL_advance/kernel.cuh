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

//#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>
//#include <gunrock/oprtr/cull_filter/cta.cuh>

#include <gunrock/oprtr/LB_CULL_advance/kernel_policy.cuh>
#include <gunrock/oprtr/advance/advance_base.cuh>

namespace gunrock {
namespace oprtr {
namespace LB_CULL {

/**
 * Not valid for this arch (default)
 * @tparam FLAG Operator flags
 * @tparam GraphT Graph type
 * @tparam InKeyT Input keys type
 * @tparam OutKeyT Output keys type
 * @tparam VALID
 */
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename ValueT, typename LabelT,
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
template <OprtrFlag FLAG, typename GraphT, typename InKeyT, typename OutKeyT,
          typename ValueT, typename LabelT>
struct Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT, true> {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  // typedef typename GraphT::ValueT           ValueT;

  typedef KernelPolicy<
      FLAG,                                             // Operator flags
      VertexT, InKeyT, OutKeyT, SizeT, ValueT, LabelT,  // Data types
      1,                                                // MAX_CTA_OCCUPANCY
      10,                                               // LOG_THREADS
      9,                                                // LOG_BLOCKS
      128 * 1024  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
      >
      KernelPolicyT;

  typedef typename KernelPolicyT::BlockScanT BlockScanT;
  // typedef typename Problem::MaskT                 MaskT;
  // typedef typename KernelPolicy::BlockLoadT       BlockLoadT;

  /*static __device__ __forceinline__ SizeT GetNeighborListLength(
      SizeT     *&d_row_offsets,
      VertexT  *&d_column_indices,
      VertexT   &d_vertex_id,
      SizeT      &max_vertex,
      SizeT      &max_edge)
      //gunrock::oprtr::advance::TYPE &ADVANCE_TYPE)
  {
      SizeT first  = LoadRowOffset<VertexId, SizeT>::Load(d_row_offsets,
  d_vertex_id);
          //(d_vertex_id >= max_vertex) ?
          max_edge : //d_row_offsets[d_vertex_id];
          //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,
  d_vertex_id);
          //_ldg(d_row_offsets + (d_vertex_id));
      SizeT second = LoadRowOffset<VertexId, SizeT>::Load(d_row_offsets,
  d_vertex_id + 1);
          //(d_vertex_id + 1 >= max_vertex) ?
          //max_edge : //d_row_offsets[d_vertex_id+1];
          //tex1Dfetch(gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,
  d_vertex_id + 1);
          //_ldg(d_row_offsets + (d_vertex_id+1));

      //printf(" d_vertex_id = %d, max_vertex = %d, max_edge = %d, first = %d,
  second = %d\n",
      //       d_vertex_id, max_vertex, max_edge, first, second);
      //return (second > first) ? second - first : 0;
      return second - first;
  }

  static __device__ __forceinline__ void GetEdgeCounts(
      SizeT    *&d_row_offsets,
      VertexT *&d_column_indices,
      SizeT    *&d_column_offsets,
      VertexT *&d_row_indices,
      VertexT *&d_queue,
      SizeT    *&d_scanned_edges,
      SizeT     &num_elements,
      SizeT     &max_vertex,
      SizeT     &max_edge,
      //gunrock::oprtr::advance::TYPE &ADVANCE_TYPE,
      bool      &in_inv,
      bool      &out_inv)
  {
      //int tid = threadIdx.x;
      //int bid = blockIdx.x;

      VertexT thread_pos = (VertexT)blockIdx.x * blockDim.x + threadIdx.x;
      if (thread_pos > num_elements)// || my_id >= max_edge)
          return;

      VertexT v_id;
      //printf("in_inv:%d, my_id:%d, column_idx:%d\n", in_inv, my_id,
  column_index); if (thread_pos < num_elements)
      {
          //if (ADVANCE_TYPE == gunrock::oprtr::advance::V2E ||
          //    ADVANCE_TYPE == gunrock::oprtr::advance::V2V)
          if ((FLAG & OprtrType_V2V) != 0 ||
              (FLAG & OprtrType_V2E) != 0)
          {
              v_id = d_queue[thread_pos];
          } else {
              v_id = (in_inv) ?
                  d_row_indices[thread_pos] : d_column_indices[thread_pos];
          }
      } else v_id = util::PreDefinedValues<VertexT>::InvalidValue;
      if (v_id < 0 || v_id > max_vertex)
      {
          d_scanned_edges[thread_pos] = 0;
          return;
      }

      //printf("my_id:%d, vid bef:%d\n", my_id, v_id);
      // add a zero length neighbor list to the end (this for getting both
  exclusive and inclusive scan in one array)
      //SizeT ncount = (!out_inv) ?
      d_scanned_edges[thread_pos] = (!out_inv) ?
          GetNeighborListLength(d_row_offsets, d_column_indices, v_id,
  max_vertex, max_edge): GetNeighborListLength(d_column_offsets, d_row_indices,
  v_id, max_vertex, max_edge);
      //printf("my_id:%d, out_inv:%d, vid:%d, ncount:%d\n", my_id, out_inv,
  v_id, ncount);
      //SizeT num_edges = (thread_pos == num_elements) ? 0 : ncount;
      //printf("%d, %d, %dG\t", my_id, v_id, num_edges);
      //d_scanned_edges[thread_pos] = num_edges;
  }

  static __device__ __forceinline__ void MarkPartitionSizes(
      SizeT        *&d_needles,
      SizeT         &split_val,
      SizeT         &num_elements,
      SizeT         &output_queue_length)
  {
      VertexT thread_pos = (VertexT) blockIdx.x * blockDim.x + threadIdx.x;
      if (thread_pos >= num_elements) return;
      SizeT try_output = (SizeT) split_val * thread_pos;
      d_needles[thread_pos] = (try_output > output_queue_length) ?
          output_queue_length : try_output;
  }*/

  static __device__ __forceinline__ void Write_Global_Output(
      typename KernelPolicyT::SmemStorage &smem_storage, SizeT &output_pos,
      SizeT &thread_output_count,
      // LabelT                    &label,
      OutKeyT *&keys_out) {
    /*if (threadIdx.x == 0)
        smem_storage.block_count = 0;
    __syncthreads();
    if (thread_output_count != 0)
        output_pos = atomicAdd(&smem_storage.block_count, thread_output_count);
    __syncthreads();
    if (threadIdx.x == 0)
        smem_storage.block_offset = atomicAdd(smem_storage.d_output_counter,
    smem_storage.block_count);
    __syncthreads();*/
    KernelPolicyT::BlockScanT::Scan(thread_output_count, output_pos,
                                    smem_storage.scan_space);

    // if (thread_output_count != 0)
    //    printf("(%4d, %4d) writting %d outputs, offset = %d\n",
    //        blockIdx.x, threadIdx.x, thread_output_count, output_pos);

    // KernelPolicy::BlockScanT(smem_storage.cub_storage.scan_space)
    //    .ExclusiveSum(thread_output_count, output_pos);
    if (threadIdx.x + 1 == KernelPolicyT::THREADS) {
      if (output_pos + thread_output_count != 0) {
        smem_storage.block_offset = atomicAdd(smem_storage.output_counter,
                                              output_pos + thread_output_count);
        // printf("(%4d, %4d) writing %d outputs, offset = %d\n",
        //    blockIdx.x, threadIdx.x, output_pos + thread_output_count,
        //    smem_storage.block_offset);
      }
    }
    __syncthreads();

    if (thread_output_count != 0) {
      output_pos += smem_storage.block_offset;
      SizeT temp_pos = (threadIdx.x << KernelPolicyT::LOG_OUTPUT_PER_THREAD);
      for (int i = 0; i < thread_output_count; i++) {
        OutKeyT u = smem_storage.thread_output_vertices[temp_pos];
        if (keys_out != NULL) {
          util::io::ModifiedStore<QUEUE_WRITE_MODIFIER>::St(
              u, keys_out + output_pos);
        }

        output_pos++;
        temp_pos++;
      }
      thread_output_count = 0;
    }
  }

  template <typename AdvanceOpT, typename FilterOpT>
  static __device__ __forceinline__ void RelaxPartitionedEdges2(
      const GraphT &graph, const VertexT &queue_index, const InKeyT *&keys_in,
      const SizeT &num_inputs, const LabelT &label, LabelT *&labels,
      unsigned char *&visited_masks, const SizeT *&output_offsets,
      const SizeT *&block_input_starts,
      // const SizeT     &partition_size,
      // const SizeT     &num_partitions,
      OutKeyT *&keys_out, ValueT *&values_out, const SizeT &num_outputs,
      // const ValueT   *&reduce_values_in,
      //      ValueT   *&reduce_values_out,
      util::CtaWorkProgress<SizeT> &work_progress, AdvanceOpT advance_op,
      FilterOpT filter_op) {
    // PrepareQueue(queue_reset, queue_index, input_queue_len, output_queue_len,
    // work_progress);
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;

    if (threadIdx.x == 0) {
      smem_storage.Init(queue_index, num_inputs, block_input_starts,
                        visited_masks, labels, output_offsets, num_outputs,
                        work_progress);
      /*printf("(%4d, %4d) : block_input = %6d ~ %6d, %6d, block_output = %9d ~
         %9d, %9d\n", blockIdx.x, threadIdx.x, smem_storage.block_input_start,
          smem_storage.block_input_end,
          smem_storage.block_input_end - smem_storage.block_input_start,
          smem_storage.block_output_start,
          smem_storage.block_output_end,
          smem_storage.block_output_size);*/
    }
    __syncthreads();
    if (smem_storage.block_output_start >= num_outputs) return;

    SizeT block_output_processed = 0;
    SizeT thread_output_count = 0;

    while (block_output_processed < smem_storage.block_output_size &&
           smem_storage.iter_input_start < smem_storage.block_input_end) {
      if (threadIdx.x == 0) {
        smem_storage.iter_input_size =
            min((SizeT)KernelPolicyT::SCRATCH_ELEMENTS - 1,
                smem_storage.block_input_end - smem_storage.iter_input_start);
        smem_storage.iter_input_end =
            smem_storage.iter_input_start + smem_storage.iter_input_size;
        smem_storage.iter_output_end =
            (smem_storage.iter_input_end < num_inputs)
                ? output_offsets[smem_storage.iter_input_end]
                : num_outputs;
        smem_storage.iter_output_end =
            min(smem_storage.iter_output_end, smem_storage.block_output_end);
        smem_storage.iter_output_size =
            min(smem_storage.iter_output_end - smem_storage.block_output_start,
                smem_storage.block_output_size);
        smem_storage.iter_output_size -= block_output_processed;
        smem_storage.iter_output_end_offset =
            smem_storage.iter_output_size + block_output_processed;

        /*printf("(%4d, %4d) : iter_input = %6d ~ %6d, %6d iter_output = %9d ~
           %9d, %9d offset = %9d ~ %9d, %9d\n", blockIdx.x, threadIdx.x,
            smem_storage.iter_input_start,
            smem_storage.iter_input_end,
            smem_storage.iter_input_size,
            smem_storage.block_output_start + block_output_processed,
            smem_storage.iter_output_end,
            smem_storage.iter_output_size,
            block_output_processed,
            smem_storage.iter_output_end_offset,
            smem_storage.iter_output_end_offset - block_output_processed);*/
      }
      __syncthreads();

      // volatile SizeT block_output_start = smem_storage.block_output_start;
      // SizeT iter_output_end_offset = smem_storage.iter_output_size +
      // block_output_processed;
      SizeT thread_input = smem_storage.iter_input_start + threadIdx.x;
      if (threadIdx.x < KernelPolicyT::SCRATCH_ELEMENTS) {
        if (thread_input <= smem_storage.block_input_end &&
            thread_input < num_inputs) {
          InKeyT input_item =
              (keys_in == NULL) ? thread_input : keys_in[thread_input];
          smem_storage.output_offset[threadIdx.x] =
              output_offsets[thread_input] - smem_storage.block_output_start;
          smem_storage.input_queue[threadIdx.x] = input_item;

          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
            // smem_storage.vertices [threadIdx.x] = input_item;
            if (util::isValid(input_item))  //(input_item >= 0)
              smem_storage.row_offset[threadIdx.x] =
                  graph.GetNeighborListOffset(input_item);
            else
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
          } else  // if (ADVANCE_TYPE == gunrock::oprtr::advance::E2V ||
                  // ADVANCE_TYPE == gunrock::oprtr::advance::E2E)
          {
            if (util::isValid(input_item)) {
              VertexT v = graph.GetEdgeDest(input_item);
              smem_storage.vertices[threadIdx.x] = v;
              smem_storage.row_offset[threadIdx.x] =
                  graph.GetNeighborListOffset(v);
            } else {
              smem_storage.vertices[threadIdx.x] =
                  util::PreDefinedValues<VertexT>::InvalidValue;
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
            }
          }

          /*if (TO_TRACK && util::pred_to_track(d_data_slice -> gpu_idx,
             input_item)) printf("(%4d, %4d) : Load Src %7d, label = %2d, "
                  "v_index = %3d, input_pos = %8d, output_offset = %8d,
             row_offset = %8d, degree = %5d" " iter_input_start = %8d,
             iter_input_end = %8d, iter_output_start_offset = %8d,
             iter_output_end_offset = %8d, block_output_start = %8d, skip_count
             = %4d\n", blockIdx.x, threadIdx.x, input_item, label-1,
             threadIdx.x, thread_input, smem_storage.output_offset[threadIdx.x],
                  smem_storage.row_offset[threadIdx.x],
                  d_row_offsets[input_item + 1] - d_row_offsets[input_item],
                  smem_storage.iter_input_start,
                  smem_storage.iter_input_end,
                  block_output_processed,
                  smem_storage.iter_output_end_offset,
                  smem_storage.block_output_start,
                  smem_storage.block_first_v_skip_count);*/
        } else {
          smem_storage.output_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;  // max_edges;
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
          } else
            smem_storage.vertices[threadIdx.x] =
                util::PreDefinedValues<VertexT>::InvalidValue;  // max_vertices;
          smem_storage.input_queue[threadIdx.x] =
              util::PreDefinedValues<VertexT>::InvalidValue;  // max_vertices;
          smem_storage.row_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;
        }
      }
      __syncthreads();

      if (threadIdx.x < KernelPolicyT::SCRATCH_ELEMENTS)
        smem_storage.row_offset[threadIdx.x] -=
            (threadIdx.x == 0) ? (block_output_processed -
                                  smem_storage.block_first_v_skip_count)
                               : smem_storage.output_offset[threadIdx.x - 1];
      __syncthreads();

      // SizeT v_index    = 0;
      VertexT v = 0;
      InKeyT input_item = 0;
      SizeT next_v_output_start_offset = 0;
      // SizeT v_output_start_offset = 0;
      SizeT row_offset_v = 0;
      SizeT v_index = 0;

      for (SizeT thread_output_offset = threadIdx.x + block_output_processed;
           thread_output_offset - threadIdx.x <
           smem_storage.iter_output_end_offset;
           thread_output_offset += KernelPolicyT::THREADS) {
        // SizeT thread_output_offset = small_iter_block_output_offset +
        // threadIdx.x; SizeT edge_id = 0; VertexId u = 0;
        bool to_process = true;
        SizeT output_pos = 0;
        unsigned char tex_mask_byte = 0;

        if (thread_output_offset < smem_storage.iter_output_end_offset) {
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
            // if (v_index > 0)
            //{
            // block_first_v_skip_count = 0;
            //    v_output_start_offset = smem_storage.output_offset[v_index-1];
            //} else v_output_start_offset = block_output_processed -
            //smem_storage.block_first_v_skip_count;
          }

          // edge_id = smem_storage.row_offset[v_index] + thread_output_offset +
          // block_first_v_skip_count - v_output_start_offset;
          SizeT edge_id = row_offset_v + thread_output_offset;
              /*+ ((v_index == 0) ? smem_storage.block_first_v_skip_count : 0)*/  //- v_output_start_offset;
          // VertexId u = (output_inverse_graph) ?
          //    d_inverse_column_indices[edge_id] :
          //    d_column_indices[edge_id];
          VertexT u = graph.GetEdgeDest(edge_id);
          // output_pos = block_output_start + thread_output_offset;
          /*if (TO_TRACK && (util::to_track(d_data_slice -> gpu_idx, u)))// ||
             util::pred_to_track(d_data_slice -> gpu_idx, v))) printf("(%4d,
             %4d) : Expand %4d, label = %d, " "src = %d, v_index = %d, edge_id =
             %d, thread_offset = %d," " iter_output_start = %d, iter_output_end
             = %d\n", blockIdx.x, threadIdx.x, u, label, v, v_index, edge_id,
             thread_output_offset, block_output_processed,
                  smem_storage.iter_output_end_offset);*/

          OutKeyT out_key = util::PreDefinedValues<OutKeyT>::InvalidValue;
          if (to_process) {
            // ProcessNeighbor
            //    <KernelPolicy, Problem, Functor,
            //    ADVANCE_TYPE, R_TYPE, R_OP>(
            //    v, u, d_data_slice, edge_id,
            //    iter_input_start + v_index, input_item,
            //    output_pos,
            //    label, d_keys_out, d_values_out,
            //    d_value_to_reduce, d_reduce_frontier);
            // SizeT input_pos = iter_input_start + v_index;
            output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
            out_key =
                ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT>(
                    v, u, edge_id, smem_storage.iter_input_start + v_index,
                    input_item, output_pos, (OutKeyT *)NULL, (ValueT *)NULL,
                    NULL, NULL,  // reduce_values_in, reduce_values_out,
                    advance_op);

            if (!util::isValid(out_key)) to_process = false;
          }

          if (to_process && (FLAG & OprtrOption_Idempotence) != 0) {
            // Location of mask byte to read
            // SizeT mask_byte_offset = (u & KernelPolicy::ELEMENT_ID_MASK) >>
            // 3;
            SizeT mask_pos = (out_key  //& KernelPolicyT::ELEMENT_ID_MASK
                              ) >>
                             3;

            // Bit in mask byte corresponding to current vertex id
            unsigned char mask_bit = 1 << (out_key & 7);
            // MaskT mask_bit = 1 << (u & ((1 << (2+sizeof(MaskT)))-1));

            // Read byte from visited mask in tex
            // tex_mask_byte = tex1Dfetch(
            //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,//cta->t_bitmask[0],
            //    output_pos);//mask_byte_offset);
            // tex_mask_byte = smem_storage.d_visited_mask[output_pos];
            tex_mask_byte = _ldg(smem_storage.visited_masks + mask_pos);

            if (!(mask_bit & tex_mask_byte)) {
              // do
              {
                tex_mask_byte |= mask_bit;
                util::io::ModifiedStore<util::io::st::cg>::St(
                    tex_mask_byte,  // mask_byte,
                    smem_storage.visited_masks + mask_pos);
                // tex_mask_byte = smem_storage.d_visited_mask[output_pos];
              }  // while (!(mask_bit & tex_mask_byte));
            } else
              to_process = false;

            if (smem_storage.labels != NULL && to_process)
            // if
            // (tex1Dfetch(gunrock::oprtr::cull_filter::LabelsTex<VertexId>::labels,
            // u) == util::MaxValue<LabelT>())
            {
              LabelT label_ = smem_storage.labels[out_key];
              if (util::isValid(label_) &&
                  label_ != util::PreDefinedValues<LabelT>::MaxValue)
                to_process = false;
              else
                smem_storage.labels[out_key] = label;
            }
          }  // else to_process = true;

          if (to_process) {
            output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
            if (!filter_op(
                    v, out_key, edge_id, out_key,
                    output_pos,  // util::PreDefinedValues<SizeT>::InvalidValue,
                    output_pos))
              to_process = false;
            if (!util::isValid(out_key)) to_process = false;
          }

          if (to_process) {
            output_pos = (threadIdx.x << KernelPolicyT::LOG_OUTPUT_PER_THREAD) +
                         thread_output_count;
            smem_storage.thread_output_vertices[output_pos] = out_key;
            //(ADVANCE_TYPE == gunrock::oprtr::advance::V2E ||
            // ADVANCE_TYPE == gunrock::oprtr::advance::E2E) ?
            // ((VertexId) edge_id) : u;
            // smem_storage.tex_mask_bytes[output_pos] = tex_mask_byte;
            thread_output_count++;
          }
        }

        if (__syncthreads_or(thread_output_count ==
                             KernelPolicyT::OUTPUT_PER_THREAD)) {
          Write_Global_Output(smem_storage, output_pos, thread_output_count,
                              keys_out);
        }
      }  // for

      __syncthreads();
      block_output_processed += smem_storage.iter_output_size;

      if (threadIdx.x == 0) {
        smem_storage.block_first_v_skip_count = 0;
        smem_storage.iter_input_start += smem_storage.iter_input_size;
      }
      __syncthreads();
    }

    if (__syncthreads_or(thread_output_count != 0)) {
      SizeT output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
      Write_Global_Output(smem_storage, output_pos, thread_output_count,
                          keys_out);
    }
  }

  template <typename AdvanceOpT, typename FilterOpT>
  static __device__ __forceinline__ void RelaxLightEdges(
      const GraphT &graph, const VertexT &queue_index, const InKeyT *&keys_in,
      const SizeT &num_inputs, const LabelT &label, LabelT *&labels,
      unsigned char *&visited_masks, const SizeT *&output_offsets,
      OutKeyT *&keys_out, ValueT *&values_out, const SizeT &num_outputs,
      // const ValueT   *&reduce_values_in,
      //      ValueT   *&reduce_values_out,
      util::CtaWorkProgress<SizeT> &work_progress, AdvanceOpT advance_op,
      FilterOpT filter_op)
  // SizeT                   *&d_output_queue_length,
  {
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;

    if (threadIdx.x == 0) {
      /*smem_storage.d_output_counter = work_progress.template
      GetQueueCounter<VertexId>(queue_index + 1);
      //int lane_id = threadIdx.x & KernelPolicy::WARP_SIZE_MASK;
      //int warp_id = threadIdx.x >> KernelPolicy::LOG_WARP_SIZE;
      smem_storage.d_labels = d_data_slice -> labels.GetPointer(util::DEVICE);
      smem_storage.d_visited_mask = d_data_slice ->
      visited_mask.GetPointer(util::DEVICE);
      printf("(%4d, %4d)\n", blockIdx.x, threadIdx.x);*/
      const SizeT *block_input_starts = NULL;
      smem_storage.Init(queue_index, num_inputs, block_input_starts,
                        visited_masks, labels, output_offsets, num_outputs,
                        work_progress);
    }
    __syncthreads();

    // SizeT* row_offsets = (output_inverse_graph) ?
    //    d_inverse_row_offsets :
    //    d_row_offsets         ;
    // VertexT* column_indices = (output_inverse_graph)?
    //    d_inverse_column_indices:
    //    d_column_indices        ;

    // SizeT partition_start    = (long long)input_queue_length * blockIdx.x /
    // gridDim.x; SizeT partition_end      = (long long)input_queue_length *
    // (blockIdx.x + 1) / gridDim.x;
    InKeyT input_item = 0;
    SizeT block_input_start =
        (SizeT)blockIdx.x * KernelPolicyT::SCRATCH_ELEMENTS;
    SizeT thread_output_count = 0;

    // while (block_input_start < partition_end)
    {
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
        if (thread_input <= block_input_end + 1 &&
            thread_input < num_inputs)  // input_queue_length)
        {
          input_item = (keys_in == NULL) ? thread_input : keys_in[thread_input];
          // printf("%d input_item = queue[%d] = %d\n",
          //    threadIdx.x, thread_input, input_item);
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
                  util::PreDefinedValues<VertexT>::InvalidValue;
              smem_storage.row_offset[threadIdx.x] =
                  util::PreDefinedValues<SizeT>::MaxValue;
            }
          }

          // printf("%d\t %d\t (%4d, %4d) : %d, %d ~ %d, %d\n",
          //    d_data_slice -> gpu_idx, label, blockIdx.x, threadIdx.x,
          //    smem_storage.vertices[threadIdx.x],
          //    smem_storage.row_offset[threadIdx.x],
          //    d_row_offsets[input_item + 1],
          //    smem_storage.output_offset[threadIdx.x]);
        }  // end of if thread_input < input_queue_length
        else {
          smem_storage.output_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;  // max_edges;
          if ((FLAG & OprtrType_V2V) != 0 || (FLAG & OprtrType_V2E) != 0) {
          } else
            smem_storage.vertices[threadIdx.x] =
                util::PreDefinedValues<VertexT>::InvalidValue;  // max_vertices;
          smem_storage.input_queue[threadIdx.x] =
              util::PreDefinedValues<InKeyT>::InvalidValue;  // max_vertices;
          smem_storage.row_offset[threadIdx.x] =
              util::PreDefinedValues<SizeT>::MaxValue;
        }
      }
      __syncthreads();

      // printf("(%4d, %4d)  : block_input = %d ~ %d, block_output = %d ~ %d,
      // %d\n",
      //    blockIdx.x, threadIdx.x,
      //    block_input_start, block_input_end,
      //    block_output_start, block_output_end, block_output_size);

      SizeT v_index = 0;
      VertexT v = 0;
      SizeT next_v_output_start_offset = 0;
      SizeT v_output_start_offset = 0;
      SizeT row_offset_v = 0;

      for (SizeT thread_output = threadIdx.x;
           thread_output - threadIdx.x < block_output_size;
           thread_output += KernelPolicyT::THREADS) {
        bool to_process = true;
        SizeT output_pos = 0;
        unsigned char tex_mask_byte = 0;
        if (thread_output < block_output_size) {
          if (thread_output >= next_v_output_start_offset) {
            v_index = util::BinarySearch<KernelPolicyT::SCRATCH_ELEMENTS>(
                thread_output, smem_storage.output_offset);
            // v          = smem_storage.vertices   [v_index];
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
          // ProcessNeighbor<KernelPolicy, Problem, Functor,
          //    ADVANCE_TYPE, R_TYPE, R_OP>(
          //    v, u, d_data_slice, edge_id,
          //    block_input_start + v_index, input_item,
          //    block_output_start + thread_output,
          //    label, d_keys_out, d_values_out,
          //    d_value_to_reduce, d_reduce_frontier);
          // printf("%d\t %d\t (%4d, %4d) : %d -> %d\n",
          //    d_data_slice -> gpu_idx, label, blockIdx.x, threadIdx.x, v, u);

          OutKeyT out_key = util::PreDefinedValues<OutKeyT>::InvalidValue;
          if (to_process) {
            output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
            out_key =
                ProcessNeighbor<FLAG, VertexT, InKeyT, OutKeyT, SizeT, ValueT>(
                    v, u, edge_id, block_input_start + v_index, input_item,
                    output_pos, (OutKeyT *)NULL, (ValueT *)NULL, NULL,
                    NULL,  // reduce_values_in, reduce_values_out,
                    advance_op);
            if (!util::isValid(out_key)) to_process = false;
          }

          if (to_process && (FLAG & OprtrOption_Idempotence) != 0) {
            SizeT mask_pos = (out_key  //& KernelPolicyT::ELEMENT_ID_MASK
                              ) >>
                             3;
            // output_pos = (u & KernelPolicyT::ELEMENT_ID_MASK) >> (2 +
            // sizeof(MaskT));

            // Bit in mask byte corresponding to current vertex id
            unsigned char mask_bit = 1 << (out_key & 7);
            // MaskT mask_bit = 1 << (u & ((1 << (2 + sizeof(MaskT)))-1));

            // Read byte from visited mask in tex
            // tex_mask_byte = tex1Dfetch(
            //    gunrock::oprtr::cull_filter::BitmaskTex<MaskT>::ref,//cta->t_bitmask[0],
            //    output_pos);//mask_byte_offset);
            tex_mask_byte = smem_storage.visited_masks[mask_pos];

            if (!(mask_bit & tex_mask_byte)) {
              do {
                tex_mask_byte |= mask_bit;
                util::io::ModifiedStore<util::io::st::cg>::St(
                    tex_mask_byte,  // mask_byte,
                    smem_storage.visited_masks + mask_pos);
                tex_mask_byte = smem_storage.visited_masks[mask_pos];
              } while (!(mask_bit & tex_mask_byte));
            } else
              to_process = false;

            if (smem_storage.labels != NULL && to_process) {
              LabelT label_ = smem_storage.labels[out_key];
              if (util::isValid(label_) &&
                  label_ != util::PreDefinedValues<LabelT>::MaxValue)
                to_process = false;
              else
                smem_storage.labels[out_key] = label;
            }
          }

          if (to_process) {
            /*if (Functor::CondFilter(
                v, u, d_data_slice, input_item, label,
                util::InvalidValue<SizeT>(),//smem_storage.iter_input_start +
            v_index, //input_pos, output_pos))
            {
                Functor::ApplyFilter(
                    v, u, d_data_slice, input_item, label,
                    util::InvalidValue<SizeT>(),//smem_storage.iter_input_start
            + v_index, //input_pos, output_pos); } else to_process = false;*/
            output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
            if (!filter_op(
                    v, out_key, edge_id, out_key,
                    output_pos,  // util::PreDefinedValues<SizeT>::InvalidValue,
                    output_pos))
              to_process = false;
            if (!util::isValid(out_key)) to_process = false;
          }

          if (to_process) {
            output_pos = (threadIdx.x << KernelPolicyT::LOG_OUTPUT_PER_THREAD) +
                         thread_output_count;
            smem_storage.thread_output_vertices[output_pos] = out_key;
            // smem_storage.tex_mask_bytes[output_pos] = tex_mask_byte;
            thread_output_count++;
          }
        }

        if (__syncthreads_or(thread_output_count ==
                             KernelPolicyT::OUTPUT_PER_THREAD)) {
          Write_Global_Output(smem_storage, output_pos, thread_output_count,
                              keys_out);
        }
      }  // end of for thread_output

      // block_input_start += KernelPolicy::SCRATCH_ELEMENTS;
      //__syncthreads();
    }

    if (__syncthreads_or(thread_output_count != 0)) {
      SizeT output_pos = util::PreDefinedValues<SizeT>::InvalidValue;
      Write_Global_Output(smem_storage, output_pos, thread_output_count,
                          keys_out);
    }
  }  // end of RelaxLightEdges
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
          typename ValueT, typename LabelT, typename AdvanceOpT,
          typename FilterOpT>
__launch_bounds__(Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT,
                           true>::KernelPolicyT::THREADS,
                  Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT,
                           true>::KernelPolicyT::CTA_OCCUPANCY) __global__
    void RelaxPartitionedEdges2(
        const bool queue_reset, const typename GraphT::VertexT queue_index,
        const GraphT graph, const InKeyT *keys_in,
        typename GraphT::SizeT num_inputs, const LabelT label, LabelT *labels,
        unsigned char *visited_masks,
        const typename GraphT::SizeT *output_offsets,
        const typename GraphT::SizeT *block_input_starts,
        // const typename GraphT::SizeT    partition_size,
        // const typename GraphT::SizeT    num_partitions,
        OutKeyT *keys_out, ValueT *values_out,
        typename GraphT::SizeT *num_outputs,
        util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
        // util::KernelRuntimeStats        kernel_stats,
        // const                  ValueT  *reduce_values_in ,
        //                       ValueT  *reduce_values_out,
        AdvanceOpT advance_op, FilterOpT filter_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs, work_progress,
               true);

  Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT>::
      RelaxPartitionedEdges2(
          graph, queue_index, keys_in, num_inputs, label, labels, visited_masks,
          output_offsets,
          block_input_starts,  // partition_size, //num_partitions,
          keys_out, values_out, num_outputs[0],
          // reduce_values_in, reduce_values_out,
          work_progress, advance_op, filter_op);
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
          typename ValueT, typename LabelT, typename AdvanceOpT,
          typename FilterOpT>
__launch_bounds__(Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT,
                           true>::KernelPolicyT::THREADS,
                  Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT,
                           true>::KernelPolicyT::CTA_OCCUPANCY) __global__
    void RelaxLightEdges(
        const bool queue_reset, const typename GraphT::VertexT queue_index,
        const GraphT graph, const InKeyT *keys_in,
        typename GraphT::SizeT num_inputs, const LabelT label, LabelT *labels,
        unsigned char *visited_masks,
        const typename GraphT::SizeT *output_offsets, OutKeyT *keys_out,
        ValueT *values_out, typename GraphT::SizeT *num_outputs,
        util::CtaWorkProgress<typename GraphT::SizeT> work_progress,
        // util::KernelRuntimeStats        kernel_stats,
        // const                  ValueT  *reduce_values_in,
        //                       ValueT  *reduce_values_out,
        AdvanceOpT advance_op, FilterOpT filter_op) {
  PrepareQueue(queue_reset, queue_index, num_inputs, num_outputs, work_progress,
               true);

  Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT>::RelaxLightEdges(
      graph, queue_index, keys_in, num_inputs, label, labels, visited_masks,
      output_offsets, keys_out, values_out, num_outputs[0],
      // reduce_values_in, reduce_values_out,
      work_progress, advance_op, filter_op);
}

/**
 * @brief Kernel entry for computing neighbor list length for each vertex in the
 * current frontier
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offsets     Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices
 * queue
 * @param[in] d_queue           Device pointer of VertexId to the incoming
 * frontier queue
 * @param[out] d_scanned_edges  Device pointer of scanned neighbor list queue of
 * the current frontier
 * @param[in] num_elements      Length of the current frontier queue
 * @param[in] max_vertices      Maximum number of elements we can place into the
 * incoming frontier
 * @param[in] max_edges         Maximum number of elements we can place into the
 * outgoing frontier
 * @param[in] ADVANCE_TYPE      Enumerator which shows the advance type: V2V,
 * V2E, E2V, or E2E
 * @param[in] in_inv            Input inverse.
 * @param[in] our_inv           Output inverse.
 */
/*template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE =
gunrock::oprtr::advance::V2V, gunrock::oprtr::advance::REDUCE_TYPE R_TYPE =
gunrock::oprtr::advance::EMPTY, gunrock::oprtr::advance::REDUCE_OP   R_OP =
gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void GetEdgeCounts(
    typename KernelPolicy::SizeT    *d_row_offsets,
    typename KernelPolicy::VertexId *d_column_indices,
    typename KernelPolicy::SizeT    *d_column_offsets,
    typename KernelPolicy::VertexId *d_row_indices,
    typename KernelPolicy::VertexId *d_queue,
    typename KernelPolicy::SizeT    *d_scanned_edges,
    typename KernelPolicy::SizeT     num_elements,
    typename KernelPolicy::SizeT     max_vertex,
    typename KernelPolicy::SizeT     max_edge,
    //gunrock::oprtr::advance::TYPE    ADVANCE_TYPE,
    bool                             in_inv,
    bool                             out_inv)
{
    Dispatch<KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_TYPE,
R_OP>::GetEdgeCounts( d_row_offsets, d_column_indices, d_column_offsets,
        d_row_indices,
        d_queue,
        d_scanned_edges,
        num_elements,
        max_vertex,
        max_edge,
        //ADVANCE_TYPE,
        in_inv,
        out_inv);
}*/

/*
 * @brief Mark partition size function.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_needles
 * @param[in] split_val
 * @param[in] num_elements Number of elements.
 * @param[in] output_queue_len Output frontier queue length.
 */
/*template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE =
gunrock::oprtr::advance::V2V, gunrock::oprtr::advance::REDUCE_TYPE R_TYPE =
gunrock::oprtr::advance::EMPTY, gunrock::oprtr::advance::REDUCE_OP   R_OP =
gunrock::oprtr::advance::NONE>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void MarkPartitionSizes(
    typename KernelPolicy::SizeT *d_needles,
    typename KernelPolicy::SizeT  split_val,
    typename KernelPolicy::SizeT  num_elements,
    typename KernelPolicy::SizeT  output_queue_len)
{
    Dispatch<KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_TYPE,
R_OP>::MarkPartitionSizes( d_needles, split_val, num_elements,
        output_queue_len);
}*/

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch_CSR_CSC(const GraphT &graph, const FrontierInT *frontier_in,
                           FrontierOutT *frontier_out, ParametersT &parameters,
                           AdvanceOpT advance_op, FilterOpT filter_op) {
  typedef typename ParametersT ::SizeT SizeT;
  typedef typename ParametersT ::ValueT ValueT;
  typedef typename ParametersT ::LabelT LabelT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT,
                            true>::KernelPolicyT KernelPolicyT;

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
    RelaxLightEdges<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT>
        <<<num_blocks, KernelPolicyT::THREADS, 0, parameters.stream>>>(
            parameters.frontier->queue_reset, parameters.frontier->queue_index,
            graph,
            (frontier_in == NULL) ? ((InKeyT *)NULL)
                                  : frontier_in->GetPointer(util::DEVICE),
            parameters.frontier->queue_length, parameters.label,
            (parameters.labels == NULL)
                ? ((LabelT *)NULL)
                : (parameters.labels->GetPointer(util::DEVICE)),
            (parameters.visited_masks == NULL)
                ? ((unsigned char *)NULL)
                : (parameters.visited_masks->GetPointer(util::DEVICE)),
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
            advance_op, filter_op);
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

    RelaxPartitionedEdges2<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT>
        <<<num_blocks, KernelPolicyT::THREADS, 0, parameters.stream>>>(
            parameters.frontier->queue_reset, parameters.frontier->queue_index,
            graph,
            (frontier_in == NULL) ? ((InKeyT *)NULL)
                                  : frontier_in->GetPointer(util::DEVICE),
            parameters.frontier->queue_length, parameters.label,
            (parameters.labels == NULL)
                ? ((LabelT *)NULL)
                : (parameters.labels->GetPointer(util::DEVICE)),
            (parameters.visited_masks == NULL)
                ? ((unsigned char *)NULL)
                : (parameters.visited_masks->GetPointer(util::DEVICE)),
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
            advance_op, filter_op);
  }

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
  typedef typename ParametersT ::SizeT SizeT;
  typedef typename ParametersT ::ValueT ValueT;
  typedef typename ParametersT ::LabelT LabelT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename Dispatch<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT,
                            true>::KernelPolicyT KernelPolicyT;

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
  RelaxLightEdges<FLAG, GraphT, InKeyT, OutKeyT, ValueT, LabelT>
      <<<num_block, KernelPolicyT::THREADS, 0, parameters.stream>>>(
          parameters.frontier->queue_reset, parameters.frontier->queue_index,
          graph,
          (frontier_in == NULL) ? ((InKeyT *)NULL)
                                : frontier_in->GetPointer(util::DEVICE),
          parameters.frontier->queue_length, parameters.label,
          (parameters.labels == NULL)
              ? ((LabelT *)NULL)
              : (parameters.labels->GetPointer(util::DEVICE)),
          (parameters.visited_masks == NULL)
              ? ((unsigned char *)NULL)
              : (parameters.visited_masks->GetPointer(util::DEVICE)),
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
          advance_op, filter_op);

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
        "LB_CULL is not implemented for given graph representation.");
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
        "LB_CULL_Light is not implemented for given graph representation.");
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
    retval = util::GRError(
        cudaErrorInvalidDeviceFunction,
        "LB_CULL is not implemented for given graph representation.");
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
    retval = util::GRError(
        cudaErrorInvalidDeviceFunction,
        "LB_CULL is not implemented for given graph representation.");
  return retval;
}

}  // namespace LB_CULL
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
