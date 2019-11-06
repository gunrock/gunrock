// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cc_enactor.cuh
 *
 * @brief CC Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/scan/block_scan.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>

namespace gunrock {
namespace app {
namespace cc {

template <typename VertexId, typename SizeT>
__global__ void Expand_Incoming_Kernel(
    const int thread_num, const SizeT num_elements,
    const VertexId *const keys_in, const VertexId *const vertex_associate_in,
    VertexId *vertex_associate_org) {
  SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

  while (x < num_elements) {
    VertexId key = _ldg(keys_in + x);
    VertexId new_pred = _ldg(vertex_associate_in + x);
    VertexId old_pred = _ldg(vertex_associate_org + key);
    if (new_pred != old_pred) {
      if (new_pred < old_pred)
        vertex_associate_org[old_pred] = new_pred;
      else if (old_pred < vertex_associate_org[new_pred])
        vertex_associate_org[new_pred] = old_pred;
    }
    // atomicMin(vertex_associate_org + old_pred, new_pred);
    // atomicMin(vertex_associate_org + new_pred, old_pred);
    // atomicMin(vertex_associate_org + key, new_pred);
    // if (TO_TRACK)
    //{
    //    if (to_track(key) || to_track(old_pred))
    //        printf("%d\t Expand_Incoming_Kernel : %d -> %d -> %d, in_pos =
    //        %d\n",
    //            thread_num, key, old_pred, new_pred, x);
    //}
    // if (new_pred < old_pred)
    //{
    // vertex_associate_org[key] = new_pred;
    //    vertex_associate_org[old_pred] = new_pred;
    // atomicMin(vertex_associate_org + old_pred, new_pred);
    //}
    x += STRIDE;
  }
}

template <typename VertexId, typename SizeT>
__global__ void First_Expand_Incoming_Kernel(
    const int thread_num, const int num_gpus, const SizeT nodes,
    VertexId **component_id_ins, VertexId *component_ids, VertexId *old_c_ids) {
  SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  __shared__ VertexId *s_component_id_ins[8];
  if (threadIdx.x < num_gpus)
    s_component_id_ins[threadIdx.x] = component_id_ins[threadIdx.x];
  __syncthreads();

  VertexId pred_ins[8], min_pred;
  while (x < nodes) {
    pred_ins[0] = /*old_c_ids[x];*/ component_ids[x];
    min_pred = pred_ins[0];
    for (int gpu = 1; gpu < num_gpus; gpu++) {
      pred_ins[gpu] = s_component_id_ins[gpu][x];
      if (pred_ins[gpu] < min_pred) min_pred = pred_ins[gpu];
    }

    // if (min_pred < component_ids[x]) component_ids[x] = min_pred;
    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (min_pred < component_ids[pred_ins[gpu]])
        component_ids[pred_ins[gpu]] = min_pred;
    //    atomicMin(component_ids + pred_ins[gpu], min_pred);
    old_c_ids[x] = pred_ins[0];
    x += STRIDE;
  }
}

template <typename KernelPolicy, typename VertexId, typename SizeT>
__global__ void Make_Output_Kernel(int thread_num, const SizeT num_vertices,
                                   VertexId *old_component_ids,
                                   VertexId *component_ids,
                                   SizeT *output_length, VertexId *keys_out,
                                   VertexId *component_out) {
  SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH,
                           KernelPolicy::LOG_THREADS>
      BlockScanT;
  __shared__ typename BlockScanT::Temp_Space scan_space;
  __shared__ SizeT block_offset;

  while (x - threadIdx.x < num_vertices) {
    bool to_process = true;
    VertexId old_cid = 0, new_cid = 0, min_cid = 0;
    if (x < num_vertices) {
      old_cid = _ldg(old_component_ids + x);
      new_cid = _ldg(component_ids + x);
      min_cid = min(new_cid, old_cid);
      if (old_cid == min_cid)
        to_process = false;
      else {
        old_component_ids[x] = min_cid;
        VertexId old_grandparent = _ldg(component_ids + old_cid);
        if (min_cid != old_grandparent) {
          // printf("%d\t Make_Output : not updated, old_cid = %d, min_cid = %d,
          // old_grandparent = %d\n",
          //    thread_num, old_cid, min_cid, old_grandparent);
          if (min_cid < old_grandparent) {
            component_ids[old_cid] = min_cid;
            old_component_ids[old_cid] = util::InvalidValue<VertexId>();
          } else {
            component_ids[min_cid] = old_grandparent;
            old_component_ids[min_cid] = util::InvalidValue<VertexId>();
          }
        }
      }
    } else
      to_process = false;

    SizeT output_pos = 0;
    BlockScanT::LogicScan(to_process, output_pos, scan_space);
    if (threadIdx.x == blockDim.x - 1) {
      if (output_pos != 0 || to_process)
        block_offset =
            atomicAdd(output_length, output_pos + ((to_process) ? 1 : 0));
    }
    __syncthreads();

    if (to_process) {
      output_pos += block_offset - 1;
      keys_out[output_pos] = x;
      component_out[output_pos] = min_cid;
      // if (TO_TRACK)
      //{
      //    if (to_track(x) || to_track(old_cid) || to_track(new_cid))
      //        printf("%d\t Make_Output : %d, cid = %d -> %d -> %d, pos =
      //        %d\n",
      //            thread_num, x, old_cid, new_cid, component_ids[new_cid],
      //            output_pos);
      //}
    }
    x += STRIDE;
  }
}

/*
 * @brief Iteration structure derived from IterationBase.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor we process on.
 */
template <typename AdvanceKernelPolicy, typename FilterKernelPolicy,
          typename Enactor>
struct CCIteration
    : public IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
                           false,  // HAS_SUBQ
                           true,   // HAS_FULLQ
                           true,   // BACKWARD
                           true,   // FORWARD
                           false>  // UPDATE_PREDECESSORS
{
 public:
  typedef typename Enactor::SizeT SizeT;
  typedef typename Enactor::Value Value;
  typedef typename Enactor::VertexId VertexId;
  typedef typename Enactor::Problem Problem;
  typedef typename Problem::DataSlice DataSlice;
  typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
  typedef typename util::DoubleBuffer<VertexId, SizeT, Value> Frontier;
  typedef IterationBase<AdvanceKernelPolicy, FilterKernelPolicy, Enactor, false,
                        true, true, true, false>
      BaseIteration;
  typedef UpdateMaskFunctor<VertexId, SizeT, Value, Problem> UpdateMaskFunctor;

  typedef HookMinFunctor<VertexId, SizeT, Value, Problem> HookMinFunctor;

  typedef HookMaxFunctor<VertexId, SizeT, Value, Problem> HookMaxFunctor;

  typedef PtrJumpFunctor<VertexId, SizeT, Value, Problem> PtrJumpFunctor;

  typedef PtrJumpMaskFunctor<VertexId, SizeT, Value, Problem>
      PtrJumpMaskFunctor;

  typedef PtrJumpUnmaskFunctor<VertexId, SizeT, Value, Problem>
      PtrJumpUnmaskFunctor;

  typedef HookInitFunctor<VertexId, SizeT, Value, Problem> HookInitFunctor;

  /*
   * @brief FullQueue_Gather function.
   *
   * @param[in] thread_num Number of threads.
   * @param[in] peer_ Peer GPU index.
   * @param[in] frontier_queue Pointer to the frontier queue.
   * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] enactor_stats Pointer to the enactor statistics.
   * @param[in] data_slice Pointer to the data slice we process on.
   * @param[in] d_data_slice Pointer to the data slice on the device.
   * @param[in] graph_slice Pointer to the graph slice we process on.
   * @param[in] work_progress Pointer to the work progress class.
   * @param[in] context CudaContext for ModernGPU API.
   * @param[in] stream CUDA stream.
   */
  static void FullQueue_Gather(
      Enactor *enactor, int thread_num, int peer_, Frontier *frontier_queue,
      util::Array1D<SizeT, SizeT> *scanned_edges,
      FrontierAttribute<SizeT> *frontier_attribute,
      EnactorStats<SizeT> *enactor_stats, DataSlice *data_slice,
      DataSlice *d_data_slice, GraphSliceT *graph_slice,
      util::CtaWorkProgressLifetime<SizeT> *work_progress, ContextPtr context,
      cudaStream_t stream) {
    if (data_slice->turn == 0) {
      frontier_attribute->queue_index = 0;
      frontier_attribute->selector = 0;
      // if (AdvanceKernelPolicy::ADVANCE_MODE ==
      // gunrock::oprtr::advance::ALL_EDGES)
      frontier_attribute->queue_length = graph_slice->edges;
      // else
      //    frontier_attribute -> queue_length =
      //        (data_slice -> num_gpus == 1 )? graph_slice -> nodes :
      //        data_slice -> local_vertices.GetSize();//graph_slice->edges;
      frontier_attribute->queue_reset = true;

      /*bool over_sized = false;
      if ((enactor -> size_check ||
           (gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
      &&
          (!data_slice -> scanned_queue_computed))
      {
          printf("scaned, queue_sizing = %d\n", frontier_attribute ->
      queue_length);fflush(stdout); if (enactor_stats -> retval = Check_Size<
      SizeT, SizeT> ( enactor -> size_check, "scanned_edges", frontier_attribute
      -> queue_length + 1, scanned_edges, over_sized, -1, -1, -1, false))
              return;
          if (enactor_stats -> retval =
      gunrock::oprtr::advance::ComputeOutputLength <AdvanceKernelPolicy,
      Problem, HookInitFunctor, gunrock::oprtr::advance::V2V>(
              frontier_attribute,
              graph_slice -> row_offsets.GetPointer(util::DEVICE),
              graph_slice -> column_indices.GetPointer(util::DEVICE),
              (SizeT*)NULL,
              (VertexId*)NULL,
              (data_slice -> num_gpus == 1) ? (VertexId*)NULL :
                  data_slice ->
      local_vertices.GetPointer(util::DEVICE),//d_in_key_queue,
              scanned_edges->GetPointer(util::DEVICE),
              graph_slice -> nodes,
              graph_slice -> edges,
              context[0],
              stream,
              //ADVANCE_TYPE,
              true,
              false,
              false)) return;
          //frontier_attribute -> output_length.Move(
          //    util::DEVICE, util::HOST, 1, 0, stream);
          //return retval;
          data_slice -> scanned_queue_computed = true;
      }

      gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, Problem, HookInitFunctor,
      gunrock::oprtr::advance::V2V>( enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats -> iteration,
          data_slice,
          d_data_slice,
          (VertexId*)NULL,
          (bool*    )NULL,
          (bool*    )NULL,
          scanned_edges -> GetPointer(util::DEVICE),
          data_slice -> num_gpus == 1 ? (VertexId*)NULL :
              data_slice -> local_vertices.GetPointer(util::DEVICE),
          (VertexId*)NULL,
          (Value*   )NULL,
          (Value*   )NULL,
          graph_slice -> row_offsets   .GetPointer(util::DEVICE),
          graph_slice -> column_indices.GetPointer(util::DEVICE),
          (SizeT*   )NULL,
          (VertexId*)NULL,
          graph_slice -> nodes,
          graph_slice -> edges,
          work_progress[0],
          context[0],
          stream,
          false,
          false,
          false);*/
      gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                           HookInitFunctor>(
          enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
          (SizeT *)NULL,          // vertex_markers
          (unsigned char *)NULL,  // visited_mask
          (VertexId *)NULL,       // keys_in
          (VertexId *)NULL, (Value *)NULL, (Value *)NULL, graph_slice->edges,
          graph_slice->nodes, work_progress[0], context[0], stream,
          graph_slice->edges, util::MaxValue<SizeT>(),
          enactor_stats->filter_kernel_stats,
          false,   // By-Pass
          false);  // skip_marking
      if (enactor->debug &&
          (enactor_stats->retval =
               util::GRError("filter::Kernel Initial HookInit Operation failed",
                             __FILE__, __LINE__)))
        return;
      enactor_stats->edges_queued[0] +=
          graph_slice->edges;  // frontier_attribute -> queue_length;
    }

    data_slice->turn++;
    frontier_attribute->queue_length = 1;
  }

  /*
   * @brief FullQueue_Core function.
   *
   * @param[in] thread_num Number of threads.
   * @param[in] peer_ Peer GPU index.
   * @param[in] frontier_queue Pointer to the frontier queue.
   * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] enactor_stats Pointer to the enactor statistics.
   * @param[in] data_slice Pointer to the data slice we process on.
   * @param[in] d_data_slice Pointer to the data slice on the device.
   * @param[in] graph_slice Pointer to the graph slice we process on.
   * @param[in] work_progress Pointer to the work progress class.
   * @param[in] context CudaContext for ModernGPU API.
   * @param[in] stream CUDA stream.
   */
  static void FullQueue_Core(
      Enactor *enactor, int thread_num, int peer_, Frontier *frontier_queue,
      util::Array1D<SizeT, SizeT> *scanned_edges,
      FrontierAttribute<SizeT> *frontier_attribute,
      EnactorStats<SizeT> *enactor_stats, DataSlice *data_slice,
      DataSlice *d_data_slice, GraphSliceT *graph_slice,
      util::CtaWorkProgressLifetime<SizeT> *work_progress, ContextPtr context,
      cudaStream_t stream) {
    /*if (data_slice -> turn == 2 && data_slice -> num_gpus > 1 && (enactor ->
    problem -> edges / 3 > enactor -> problem -> nodes)) { // special
    expand_incoming for first data exchagne for (int peer = 1; peer < data_slice
    -> num_gpus; peer++) data_slice -> vertex_associate_ins[peer] = data_slice
    -> vertex_associate_in[enactor_stats -> iteration
    %2][peer].GetPointer(util::DEVICE); data_slice ->
    vertex_associate_ins.Move(util::HOST, util::DEVICE, data_slice -> num_gpus,
    0, stream); First_Expand_Incoming_Kernel<<<240, 512, 0, stream>>>
            (thread_num,
            data_slice -> num_gpus,
            graph_slice -> nodes,
            data_slice -> vertex_associate_ins.GetPointer(util::DEVICE),
            data_slice -> component_ids.GetPointer(util::DEVICE),
            data_slice -> old_c_ids.GetPointer(util::DEVICE));
        for (int peer = 1; peer < data_slice -> num_gpus; peer++)
        {
            data_slice -> keys_out[peer].ForceSetPointer(data_slice ->
    temp_vertex_out, util::DEVICE); data_slice ->
    vertex_associate_out[peer].ForceSetPointer(data_slice -> temp_comp_out,
    util::DEVICE);
        }
    }*/

    enactor_stats->iteration = 0;
    frontier_attribute->queue_index = 0;
    frontier_attribute->selector = 0;
    frontier_attribute->queue_length =
        /*data_slice -> turn <= 1 ?*/ graph_slice->nodes /* :
data_slice -> local_vertices.GetSize()*/
        ;
    frontier_attribute->queue_reset = true;

    // util::MemsetCopyVectorKernel <<<240, 512, 0, stream>>>(
    //    data_slice -> old_c_ids.GetPointer(util::DEVICE),
    //    data_slice -> component_ids.GetPointer(util::DEVICE),
    //    data_slice -> nodes);

    // First Pointer Jumping Round
    data_slice->vertex_flag[0] = 0;
    while (!data_slice->vertex_flag[0]) {
      data_slice->vertex_flag[0] = 1;
      data_slice->vertex_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);

      gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                           PtrJumpFunctor>(
          enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
          (SizeT *)NULL,                                 // vertex_markers,
          (unsigned char *)NULL,                         // visited_mask,
          /*data_slice -> turn <= 1 ?*/ (VertexId *)NULL /* :
                data_slice -> local_vertices.GetPointer(util::DEVICE)*/
          ,
          // frontier_queue -> values[frontier_attribute ->
          // selector].GetPointer(util::DEVICE),
          (VertexId *)NULL, (Value *)NULL, (Value *)NULL,
          /*data_slice -> turn <= 1 ?*/ graph_slice->nodes /* :
             data_slice -> local_vertices.GetSize()*/
          ,  // frontier_attribute -> output_length[0],
          graph_slice->nodes, work_progress[0], context[0], stream,
          graph_slice->nodes,  // frontier_queue -> values[frontier_attribute ->
                               // selector].GetSize(),
          util::MaxValue<SizeT>(), enactor_stats->filter_kernel_stats,
          false,   // By-Pass
          false);  // skip_marking

      if (enactor->debug &&
          (enactor_stats->retval = util::GRError(
               "filter::Kernel First Pointer Jumping Round failed", __FILE__,
               __LINE__)))
        return;
      enactor_stats->nodes_queued[0] += frontier_attribute->queue_length;

      frontier_attribute->queue_reset = false;
      frontier_attribute->queue_index++;
      enactor_stats->iteration++;
      data_slice->vertex_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);

      if (enactor_stats->retval =
              util::GRError(cudaStreamSynchronize(stream),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__))
        return;
      // Check if done
      if (data_slice->vertex_flag[0]) break;
    }

    if (data_slice->turn > 1 &&
        (enactor->problem->edges / 3 > enactor->problem->nodes)) {
      enactor_stats->iteration = data_slice->turn;
      return;
    }

    util::MemsetKernel<<<240, 512, 0, stream>>>(
        data_slice->marks.GetPointer(util::DEVICE), false, graph_slice->edges);
    frontier_attribute->queue_index = 0;  // Work queue index
    frontier_attribute->selector = 0;
    frontier_attribute->queue_length = graph_slice->nodes;
    frontier_attribute->queue_reset = true;

    gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                         UpdateMaskFunctor>(
        enactor_stats[0], frontier_attribute[0],
        (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
        (SizeT *)NULL,          // vertex_markers,
        (unsigned char *)NULL,  // visited_mask,
        (VertexId *)NULL,  // frontier_queue -> values[frontier_attribute ->
                           // selector].GetPointer(util::DEVICE),
        (VertexId *)NULL, (Value *)NULL, (Value *)NULL,
        graph_slice->nodes,  // frontier_attribute -> output_length[0],
        graph_slice->nodes, work_progress[0], context[0], stream,
        graph_slice->nodes,  // frontier_queue -> values[frontier_attribute ->
                             // selector].GetSize(),
        util::MaxValue<SizeT>(), enactor_stats->filter_kernel_stats,
        false,   // By-Pass
        false);  // skip_marking

    if (enactor->debug && (enactor_stats->retval = util::GRError(
                               "filter::Kernel Update Mask Operation failed",
                               __FILE__, __LINE__)))
      return;
    enactor_stats->nodes_queued[0] += frontier_attribute->queue_length;

    enactor_stats->iteration = 1;
    data_slice->edge_flag[0] = 0;
    while (!data_slice->edge_flag[0]) {
      frontier_attribute->queue_index = 0;  // Work queue index
      // if (AdvanceKernelPolicy::ADVANCE_MODE ==
      // gunrock::oprtr::advance::ALL_EDGES)
      frontier_attribute->queue_length = graph_slice->edges;
      // else frontier_attribute->queue_length =
      //    (data_slice -> num_gpus == 1) ? graph_slice -> nodes :
      //    data_slice -> local_vertices.GetSize();//graph_slice->edges;
      frontier_attribute->selector = 0;
      frontier_attribute->queue_reset = true;
      data_slice->edge_flag[0] = 1;
      data_slice->edge_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);

      gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                           HookMaxFunctor>(
          enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
          (SizeT *)NULL,          // vertex_markers,
          (unsigned char *)NULL,  // visited_mask,
          (VertexId *)NULL,       // keys_in
          (VertexId *)NULL, (Value *)NULL, (Value *)NULL, graph_slice->edges,
          graph_slice->nodes, work_progress[0], context[0], stream,
          graph_slice->edges, util::MaxValue<SizeT>(),
          enactor_stats->filter_kernel_stats,
          false,   // By-Pass
          false);  // skip_marking

      /*gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, Problem, HookMaxFunctor,
         gunrock::oprtr::advance::V2V>( enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats -> iteration ,
          data_slice,
          d_data_slice,
          (VertexId*)NULL,
          (bool*    )NULL,
          (bool*    )NULL,
          scanned_edges -> GetPointer(util::DEVICE),
          (data_slice -> num_gpus == 1)  ? (VertexId*)NULL :
              data_slice -> local_vertices.GetPointer(util::DEVICE),
          (VertexId*)NULL,
          (Value*   )NULL,
          (Value*   )NULL,
          graph_slice -> row_offsets   .GetPointer(util::DEVICE),
          graph_slice -> column_indices.GetPointer(util::DEVICE),
          (SizeT*   )NULL,
          (VertexId*)NULL,
          graph_slice -> nodes,
          graph_slice -> edges,
          work_progress[0],
          context[0],
          stream,
          false,
          false,
          false);*/
      //}
      if (enactor->debug && (enactor_stats->retval = util::GRError(
                                 "filter::Kernel Hook Min/Max Operation failed",
                                 __FILE__, __LINE__)))
        return;
      enactor_stats->edges_queued[0] += frontier_attribute->queue_length;

      frontier_attribute->queue_reset = false;
      frontier_attribute->queue_index++;
      enactor_stats->iteration++;

      data_slice->edge_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);
      if (enactor_stats->retval =
              util::GRError(cudaStreamSynchronize(stream),
                            "cudaStreamSynchronize failed", __FILE__, __LINE__))
        return;
      // Check if done
      if (data_slice->edge_flag[0])
        break;  //|| enactor_stats->iteration>5) break;

      ///////////////////////////////////////////
      // Pointer Jumping
      frontier_attribute->queue_index = 0;
      frontier_attribute->selector = 0;
      frontier_attribute->queue_length = graph_slice->nodes;
      frontier_attribute->queue_reset = true;

      // First Pointer Jumping Round
      data_slice->vertex_flag[0] = 0;
      while (!data_slice->vertex_flag[0]) {
        data_slice->vertex_flag[0] = 1;
        data_slice->vertex_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);

        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             PtrJumpMaskFunctor>(
            enactor_stats[0], frontier_attribute[0],
            (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
            (SizeT *)NULL,          // vertex_markers,
            (unsigned char *)NULL,  // visited_mask,
            (VertexId *)NULL,  // frontier_queue ->values[frontier_attribute ->
                               // selector].GetPointer(util::DEVICE),
            (VertexId *)NULL, (Value *)NULL, (Value *)NULL,
            graph_slice->nodes,  // frontier_attribute -> output_length[0],
            graph_slice->nodes, work_progress[0], context[0], stream,
            graph_slice->nodes,  // frontier_queue -> values[frontier_attribute
                                 // -> selector].GetSize(),
            util::MaxValue<SizeT>(), enactor_stats->filter_kernel_stats,
            false,   // By-Pass
            false);  // skip_marking

        if (enactor->debug && (enactor_stats->retval = util::GRError(
                                   "filter::Kernel Pointer Jumping Mask failed",
                                   __FILE__, __LINE__)))
          return;
        enactor_stats->nodes_queued[0] += frontier_attribute->queue_length;

        frontier_attribute->queue_reset = false;
        frontier_attribute->queue_index++;

        data_slice->vertex_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);
        if (enactor_stats->retval = util::GRError(
                cudaStreamSynchronize(stream), "cudaStreamSynchronize failed",
                __FILE__, __LINE__))
          return;
        // Check if done
        if (data_slice->vertex_flag[0]) break;
      }

      frontier_attribute->queue_index = 0;  // Work queue index
      frontier_attribute->selector = 0;
      frontier_attribute->queue_length = graph_slice->nodes;
      frontier_attribute->queue_reset = true;

      gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                           PtrJumpUnmaskFunctor>(
          enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
          (SizeT *)NULL,          // vertex_markers,
          (unsigned char *)NULL,  // visited_mask,
          (VertexId *)NULL,  // frontier_queue -> values[frontier_attribute ->
                             // selector].GetPointer(util::DEVICE),
          (VertexId *)NULL, (Value *)NULL, (Value *)NULL,
          graph_slice->nodes,  // frontier_attribute -> output_length[0],
          graph_slice->nodes, work_progress[0], context[0], stream,
          graph_slice->nodes,  // frontier_queue -> values[frontier_attribute ->
                               // selector].GetSize(),
          util::MaxValue<SizeT>(), enactor_stats->filter_kernel_stats,
          false,   // By-Pass
          false);  // skip_marking

      if (enactor->debug &&
          (enactor_stats->retval = util::GRError(
               "filter::Kernel Pointer Jumping Unmask Operation failed",
               __FILE__, __LINE__)))
        return;
      enactor_stats->nodes_queued[0] += frontier_attribute->queue_length;

      gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                           UpdateMaskFunctor>(
          enactor_stats[0], frontier_attribute[0],
          (VertexId)enactor_stats->iteration, data_slice, d_data_slice,
          (SizeT *)NULL,          // vertex_markers,
          (unsigned char *)NULL,  // visited_mask,
          (VertexId *)NULL,  // frontier_queue -> values[frontier_attribute ->
                             // selector].GetPointer(util::DEVICE),
          (VertexId *)NULL, (Value *)NULL, (Value *)NULL,
          graph_slice->nodes,  // frontier_attribute -> output_length[0],
          graph_slice->nodes, work_progress[0], context[0], stream,
          graph_slice->nodes,  // frontier_queue -> values[frontier_attribute ->
                               // selector].GetSize(),
          util::MaxValue<SizeT>(), enactor_stats->filter_kernel_stats,
          false,   // By-Pass
          false);  // skip_marking

      if (enactor->debug && (enactor_stats->retval = util::GRError(
                                 "filter::Kernel Update Mask Operation failed",
                                 __FILE__, __LINE__)))
        return;
      enactor_stats->nodes_queued[0] += frontier_attribute->queue_length;

      ///////////////////////////////////////////
    }

    enactor_stats->iteration = data_slice->turn;
  }

  /*
   * @brief Compute output queue length function.
   *
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] d_offsets Pointer to the offsets.
   * @param[in] d_indices Pointer to the indices.
   * @param[in] d_in_key_queue Pointer to the input mapping queue.
   * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
   * @param[in] max_in Maximum input queue size.
   * @param[in] max_out Maximum output queue size.
   * @param[in] context CudaContext for ModernGPU API.
   * @param[in] stream CUDA stream.
   * @param[in] ADVANCE_TYPE Advance kernel mode.
   * @param[in] express Whether or not enable express mode.
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  static cudaError_t Compute_OutputLength(
      Enactor *enactor, FrontierAttribute<SizeT> *frontier_attribute,
      // DataSlice                     *data_slice,
      // DataSlice                     *d_data_slice,
      SizeT *d_offsets, VertexId *d_indices, SizeT *d_inv_offsets,
      VertexId *d_inv_indices, VertexId *d_in_key_queue,
      util::Array1D<SizeT, SizeT> *partitioned_scanned_edges, SizeT max_in,
      SizeT max_out, CudaContext &context, cudaStream_t stream,
      gunrock::oprtr::advance::TYPE ADVANCE_TYPE, bool express = false,
      bool in_inv = false, bool out_inv = false) {
    // util::MemsetKernel<SizeT><<<1,1,0,stream>>>(
    //    frontier_attribute->output_length.GetPointer(util::DEVICE),
    //    frontier_attribute->queue_length ==0 ? 0 :
    //    1/*frontier_attribute->queue_length*/, 1);
    cudaError_t retval = cudaSuccess;
    // printf("SIZE_CHECK = %s\n", Enactor::SIZE_CHECK ? "true" : "false");
    frontier_attribute->output_length[0] =
        (frontier_attribute->queue_length == 0) ? 0 : 1;
    return retval;
  }

  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  static void Expand_Incoming(
      Enactor *enactor, cudaStream_t stream, VertexId iteration, int peer_,
      SizeT received_length, SizeT num_elements,
      util::Array1D<SizeT, SizeT> &in_length_out,
      util::Array1D<SizeT, VertexId> &keys_in,
      util::Array1D<SizeT, VertexId> &vertex_associate_in,
      util::Array1D<SizeT, Value> &value__associate_in,
      util::Array1D<SizeT, VertexId> &keys_out,
      util::Array1D<SizeT, VertexId *> &vertex_associate_orgs,
      util::Array1D<SizeT, Value *> &value__associate_orgs,
      DataSlice *h_data_slice, EnactorStats<SizeT> *enactor_stats) {
    // printf("%d\t %d\t Expand_Incoming num_elements = %d\n",
    //    h_data_slice -> gpu_idx, enactor_stats -> iteration, num_elements);
    // if (h_data_slice -> turn == 1 && (enactor -> problem -> edges / 3 >
    // enactor -> problem -> nodes)) return;
    int num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2 + 1;
    if (num_blocks > 480) num_blocks = 480;
    Expand_Incoming_Kernel<VertexId, SizeT>
        <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>(
            h_data_slice->gpu_idx, num_elements,
            keys_in.GetPointer(util::DEVICE),
            vertex_associate_in.GetPointer(util::DEVICE),
            vertex_associate_orgs[0]);
    if (!enactor->problem->unified_receive) in_length_out[peer_] = 0;
  }
  /*
   * @brief Check frontier queue size function.
   *
   * @param[in] thread_num Number of threads.
   * @param[in] peer_ Peer GPU index.
   * @param[in] request_length Request frontier queue length.
   * @param[in] frontier_queue Pointer to the frontier queue.
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] enactor_stats Pointer to the enactor statistics.
   * @param[in] graph_slice Pointer to the graph slice we process on.
   */
  static void Check_Queue_Size(Enactor *enactor, int thread_num, int peer_,
                               SizeT request_length, Frontier *frontier_queue,
                               FrontierAttribute<SizeT> *frontier_attribute,
                               EnactorStats<SizeT> *enactor_stats,
                               GraphSliceT *graph_slice) {}

  /*
   * @brief Stop_Condition check function.
   *
   * @param[in] enactor_stats Pointer to the enactor statistics.
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] data_slice Pointer to the data slice we process on.
   * @param[in] num_gpus Number of GPUs used.
   */
  static bool Stop_Condition(EnactorStats<SizeT> *enactor_stats,
                             FrontierAttribute<SizeT> *frontier_attribute,
                             util::Array1D<SizeT, DataSlice> *data_slice,
                             int num_gpus) {
    // printf("CC Stop checked\n");fflush(stdout);
    for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++)
      if (enactor_stats[gpu].retval != cudaSuccess) {
        printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval,
               gpu % num_gpus, cudaGetErrorString(enactor_stats[gpu].retval));
        fflush(stdout);
        return true;
      }

    if (num_gpus < 2 && data_slice[0]->turn > 0) return true;

    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (data_slice[gpu]->turn == 0) {
        // printf("data_slice[%d]->turn==0\n", gpu);
        // fflush(stdout);
        return false;
      }

    for (int gpu = 0; gpu < num_gpus; gpu++)
      for (int peer = 1; peer < num_gpus; peer++)
        for (int i = 0; i < 2; i++)
          if (data_slice[gpu]->in_length[i][peer] != 0) {
            // printf("data_slice[%d]->in_length[%d][%d] = %lld\n",
            //    gpu, i, peer,
            //    (long long)data_slice[gpu]->in_length[i][peer]);
            // fflush(stdout);
            return false;
          }

    for (int gpu = 0; gpu < num_gpus; gpu++)
      for (int peer = 0; peer < num_gpus; peer++)
        if (data_slice[gpu]->out_length[peer] != 0) {
          // printf("data_slice[%d] -> out_length[%d] = %lld\n",
          //    gpu, peer, (long long)data_slice[gpu]->out_length[peer]);
          // fflush(stdout);
          return false;
        }

    if (num_gpus > 1)
      for (int gpu = 0; gpu < num_gpus; gpu++)
        if (data_slice[gpu]->has_change || data_slice[gpu]->previous_change) {
          // printf("data_slice[%d] -> has_change = %s, previous_change = %s\n",
          //    gpu, data_slice[gpu] -> has_change ? "true" : "false",
          //    data_slice[gpu] -> previous_change ? "true" : "false");
          // fflush(stdout);
          return false;
        }
    // printf("CC to stop\n");fflush(stdout);
    return true;
  }

  /*
   * @brief Make_Output function.
   *
   * @tparam NUM_VERTEX_ASSOCIATES
   * @tparam NUM_VALUE__ASSOCIATES
   *
   * @param[in] thread_num Number of threads.
   * @param[in] num_elements
   * @param[in] num_gpus Number of GPUs used.
   * @param[in] frontier_queue Pointer to the frontier queue.
   * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] enactor_stats Pointer to the enactor statistics.
   * @param[in] data_slice Pointer to the data slice we process on.
   * @param[in] graph_slice Pointer to the graph slice we process on.
   * @param[in] work_progress Pointer to the work progress class.
   * @param[in] context CudaContext for ModernGPU API.
   * @param[in] stream CUDA stream.
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  static void Make_Output(Enactor *enactor, int thread_num, SizeT num_elements,
                          int num_gpus, Frontier *frontier_queue,
                          util::Array1D<SizeT, SizeT> *scanned_edges,
                          FrontierAttribute<SizeT> *frontier_attribute,
                          EnactorStats<SizeT> *enactor_stats,
                          util::Array1D<SizeT, DataSlice> *data_slice_,
                          GraphSliceT *graph_slice,
                          util::CtaWorkProgressLifetime<SizeT> *work_progress,
                          ContextPtr context, cudaStream_t stream) {
    DataSlice *data_slice = data_slice_->GetPointer(util::HOST);
    /*if (data_slice -> turn == 1 && (enactor -> problem -> edges / 3 > enactor
    -> problem -> nodes))
    {
        //util::MemsetCopyVectorKernel<<<240, 512, 0, stream>>>(
        //    data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE),
        //    data_slice -> component_ids.GetPointer(util::DEVICE),
        //    graph_slice -> nodes);
        //util::MemsetCopyVectorKernel<<<240, 512, 0, stream>>>(
        //    data_slice -> old_c_ids.GetPointer(util::DEVICE),
        //    data_slice -> component_ids.GetPointer(util::DEVICE),
        //    graph_slice -> nodes);
        //util::MemsetIdxKernel<<<240, 512, 0, stream>>>(
        //    data_slice -> keys_out[1].GetPointer(util::DEVICE),
        //    graph_slice -> nodes);
        data_slice -> temp_vertex_out = data_slice ->
    keys_out[1].GetPointer(util::DEVICE); data_slice -> temp_comp_out =
    data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE); for (int
    peer_ = 1; peer_ < num_gpus; peer_++)
        {
            data_slice -> keys_out[peer_].ForceSetPointer(NULL, util::DEVICE);
            data_slice -> vertex_associate_out[peer_].ForceSetPointer(data_slice
    -> component_ids.GetPointer(util::DEVICE), util::DEVICE);
        }
        data_slice -> out_length[1] = graph_slice -> nodes + 1;
    } else*/
    {
      data_slice->out_length[1] = 1;
      data_slice->out_length.Move(util::HOST, util::DEVICE, 1, 1, stream);
      int num_blocks = data_slice->nodes / AdvanceKernelPolicy::THREADS + 1;
      if (num_blocks > 480) num_blocks = 480;
      Make_Output_Kernel<AdvanceKernelPolicy, VertexId, SizeT>
          <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>(
              thread_num, data_slice->nodes,
              data_slice->old_c_ids.GetPointer(util::DEVICE),
              data_slice->component_ids.GetPointer(util::DEVICE),
              data_slice->out_length.GetPointer(util::DEVICE) + 1,
              data_slice->keys_out[1].GetPointer(util::DEVICE),
              data_slice->vertex_associate_out[1].GetPointer(util::DEVICE));
      data_slice->out_length.Move(util::DEVICE, util::HOST, 1, 1, stream);
      // util::MemsetCopyVectorKernel <<<240, 512, 0, stream>>>(
      //    data_slice -> old_c_ids.GetPointer(util::DEVICE),
      //    data_slice -> component_ids.GetPointer(util::DEVICE),
      //    data_slice -> nodes);
    }
    if (enactor_stats->retval =
            util::GRError(cudaStreamSynchronize(stream),
                          "cudaStreamSynchronize failed", __FILE__, __LINE__))
      return;
    // printf("%d num_diff = %d\n", thread_num, data_slice -> out_length[1]);
    data_slice->out_length[1]--;

    // printf("%d\t %lld\t changes = %lld\n",
    //    thread_num, enactor_stats -> iteration,
    //    (long long)data_slice -> out_length[1]);
    // fflush(stdout);
    data_slice->previous_change = data_slice->has_change;
    for (int i = 0; i < num_gpus; i++)
      data_slice->out_length[i] = data_slice->out_length[1];
    if (data_slice->out_length[1] != 0)
      data_slice->has_change = true;
    else
      data_slice->has_change = false;
  }

  /*
   * @brief Iteration_Update_Preds function.
   *
   * @param[in] graph_slice Pointer to the graph slice we process on.
   * @param[in] data_slice Pointer to the data slice we process on.
   * @param[in] frontier_attribute Pointer to the frontier attribute.
   * @param[in] frontier_queue Pointer to the frontier queue.
   * @param[in] num_elements Number of elements.
   * @param[in] stream CUDA stream.
   */
  static void Iteration_Update_Preds(
      Enactor *enactor, GraphSliceT *graph_slice, DataSlice *data_slice,
      FrontierAttribute<SizeT> *frontier_attribute, Frontier *frontier_queue,
      SizeT num_elements, cudaStream_t stream) {}
};

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam CcEnactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template <typename AdvanceKernelPolicy, typename FilterKernelPolicy,
          typename Enactor>
static CUT_THREADPROC CCThread(void *thread_data_) {
  typedef typename Enactor::Problem Problem;
  typedef typename Enactor::SizeT SizeT;
  typedef typename Enactor::VertexId VertexId;
  typedef typename Enactor::Value Value;
  typedef typename Problem::DataSlice DataSlice;
  typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;
  typedef UpdateMaskFunctor<VertexId, SizeT, Value, Problem> Functor;
  ThreadSlice *thread_data = (ThreadSlice *)thread_data_;
  Problem *problem = (Problem *)thread_data->problem;
  Enactor *enactor = (Enactor *)thread_data->enactor;
  int num_gpus = problem->num_gpus;
  int thread_num = thread_data->thread_num;
  int gpu_idx = problem->gpu_idx[thread_num];
  DataSlice *data_slice =
      problem->data_slices[thread_num].GetPointer(util::HOST);
  FrontierAttribute<SizeT> *frontier_attribute =
      &(enactor->frontier_attribute[thread_num * num_gpus]);
  EnactorStats<SizeT> *enactor_stats =
      &(enactor->enactor_stats[thread_num * num_gpus]);

  // printf("CCThread entered\n");fflush(stdout);
  if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) {
    thread_data->status = ThreadSlice::Status::Idle;
    CUT_THREADEND;
  }

  thread_data->status = ThreadSlice::Status::Idle;

  while (thread_data->status != ThreadSlice::Status::ToKill) {
    while (thread_data->status == ThreadSlice::Status::Wait ||
           thread_data->status == ThreadSlice::Status::Idle) {
      sleep(0);
    }
    if (thread_data->status == ThreadSlice::Status::ToKill) break;

    for (int peer_ = 0; peer_ < num_gpus; peer_++) {
      frontier_attribute[peer_].queue_index = 0;
      frontier_attribute[peer_].selector = 0;
      frontier_attribute[peer_].queue_length = 0;
      frontier_attribute[peer_].queue_reset = true;
      enactor_stats[peer_].iteration = 0;
    }
    if (num_gpus > 1) {
      data_slice->vertex_associate_orgs[0] =
          data_slice->component_ids.GetPointer(util::DEVICE);
      data_slice->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
    }

    gunrock::app::Iteration_Loop<
        Enactor, Functor,
        CCIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>, 1, 0>(
        thread_data);
    thread_data->status = ThreadSlice::Status::Idle;
  }

  // printf("CC_Thread finished\n");fflush(stdout);
  thread_data->status = ThreadSlice::Status::Ended;
  CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 *
 * @tparam _Problem Problem type we process on
 * @tparam _INSTRUMENT Whether or not to collect per-CTA clock-count stats.
 * @tparam _DEBUG Whether or not to enable debug mode.
 * @tparam _SIZE_CHECK Whether or not to enable size check.
 */
template <typename _Problem>
// bool _INSTRUMENT,
// bool _DEBUG,
// bool _SIZE_CHECK>
class CCEnactor
    : public EnactorBase<typename _Problem::SizeT /*, _DEBUG, _SIZE_CHECK*/> {
  // Members
  ThreadSlice *thread_slices;
  CUTThread *thread_Ids;

  // Methods
 public:
  _Problem *problem;
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexId VertexId;
  typedef typename Problem::Value Value;
  typedef EnactorBase<SizeT> BaseEnactor;
  typedef CCEnactor<Problem> Enactor;
  // static const bool INSTRUMENT = _INSTRUMENT;
  // static const bool DEBUG      = _DEBUG;
  // static const bool SIZE_CHECK = _SIZE_CHECK;

 public:
  /**
   * @brief CCEnactor default constructor
   */
  CCEnactor(int num_gpus = 1, int *gpu_idx = NULL, bool instrument = false,
            bool debug = false, bool size_check = true)
      : BaseEnactor(EDGE_FRONTIERS, num_gpus, gpu_idx, instrument, debug,
                    size_check) {
    thread_slices = NULL;
    thread_Ids = NULL;
    problem = NULL;
  }

  /**
   * @brief CCEnactor default destructor
   */
  virtual ~CCEnactor() { Release(); }

  cudaError_t Release() {
    cudaError_t retval = cudaSuccess;
    if (thread_slices != NULL) {
      for (int gpu = 0; gpu < this->num_gpus; gpu++)
        thread_slices[gpu].status = ThreadSlice::Status::ToKill;
      cutWaitForThreads(thread_Ids, this->num_gpus);
      delete[] thread_Ids;
      thread_Ids = NULL;
      delete[] thread_slices;
      thread_slices = NULL;
    }
    if (retval = BaseEnactor::Release()) return retval;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the problem.
   *
   * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
   * @tparam FilterKernelPolicy Kernel policy for filter operator.
   *
   * @param[in] context CudaContext pointer for ModernGPU API.
   * @param[in] problem Pointer to Problem object.
   * @param[in] max_grid_size Maximum grid size for kernel calls.
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
  cudaError_t InitCC(ContextPtr *context, Problem *problem,
                     int max_grid_size = 512) {
    cudaError_t retval = cudaSuccess;
    // Lazy initialization
    if (retval = BaseEnactor::Init(
            // problem,
            max_grid_size, AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
      return retval;

    /*for (int gpu=0;gpu<this->num_gpus;gpu++)
    {
        if (retval = util::SetDevice(this->gpu_idx[gpu])) break;
        if (sizeof(SizeT) == 4)
        {
            cudaChannelFormatDesc row_offsets_dest =
    cudaCreateChannelDesc<SizeT>();
            gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets.channelDesc
    = row_offsets_dest; if (retval = util::GRError(cudaBindTexture( 0,
                gunrock::oprtr::edge_map_partitioned::RowOffsetsTex<SizeT>::row_offsets,
                problem->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE),
                ((size_t) (problem -> graph_slices[gpu]->nodes + 1)) *
    sizeof(SizeT)), "BFSEnactor cudaBindTexture row_offsets_ref failed",
                __FILE__, __LINE__)) break;
        }
    }*/

    if (this->debug) {
      printf("CC vertex map occupancy %d, level-grid size %d\n",
             FilterKernelPolicy::CTA_OCCUPANCY,
             this->enactor_stats[0].filter_grid_size);
    }

    this->problem = problem;
    thread_slices = new ThreadSlice[this->num_gpus];
    thread_Ids = new CUTThread[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      // thread_slices[gpu].cpu_barrier  = cpu_barrier;
      thread_slices[gpu].thread_num = gpu;
      thread_slices[gpu].problem = (void *)problem;
      thread_slices[gpu].enactor = (void *)this;
      thread_slices[gpu].context = &(context[gpu * this->num_gpus]);
      thread_slices[gpu].status = ThreadSlice::Status::Inited;
      thread_slices[gpu].thread_Id =
          cutStartThread((CUT_THREADROUTINE) &
                             (CCThread<AdvanceKernelPolicy, FilterKernelPolicy,
                                       CCEnactor<Problem>>),
                         (void *)&(thread_slices[gpu]));
      thread_Ids[gpu] = thread_slices[gpu].thread_Id;
    }

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      while (thread_slices[gpu].status != ThreadSlice::Status::Idle) {
        sleep(0);
        // std::this_thread::yield();
      }
    }
    return retval;
  }

  /**
   * @brief Reset enactor
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  cudaError_t Reset() {
    cudaError_t retval = cudaSuccess;
    if (retval = BaseEnactor::Reset()) return retval;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      thread_slices[gpu].status = ThreadSlice::Status::Wait;
    }
    return retval;
  }

  /**
   * @brief Enacts a connected-component computing on the specified graph.
   *
   * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
   * @tparam FilterKernelPolicy Kernel policy for filter operator.
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
  cudaError_t EnactCC() {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      thread_slices[gpu].status = ThreadSlice::Status::Running;
    }
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      while (thread_slices[gpu].status != ThreadSlice::Status::Idle) {
        sleep(0);
        // std::this_thread::yield();
      }
    }

    for (int gpu = 0; gpu < this->num_gpus * this->num_gpus; gpu++)
      if (this->enactor_stats[gpu].retval != cudaSuccess) {
        retval = this->enactor_stats[gpu].retval;
        return retval;
      }

    if (this->debug) printf("\nGPU CC Done.\n");
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  typedef gunrock::oprtr::advance::KernelPolicy<
      Problem,   // Problem data type
      300,       // CUDA_ARCH
      8,         // MIN_CTA_OCCUPANCY,
      8,         // LOG_THREADS,
      8,         // LOG_BLOCKS,
      32 * 128,  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
      1,         // LOG_LOAD_VEC_SIZE,
      0,         // LOG_LOADS_PER_TILE
      5,         // LOG_RAKING_THREADS,
      32,        // WART_GATHER_THRESHOLD,
      128 * 4,   // CTA_GATHER_THRESHOLD,
      7,         // LOG_SCHEDULE_GRANULARITY,
      gunrock::oprtr::advance::LB>
      LB_AdvanceKernelPolicy;

  typedef gunrock::oprtr::advance::KernelPolicy<
      Problem,   // Problem data type
      300,       // CUDA_ARCH
      8,         // MIN_CTA_OCCUPANCY,
      8,         // LOG_THREADS,
      8,         // LOG_BLOCKS,
      32 * 128,  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
      1,         // LOG_LOAD_VEC_SIZE,
      0,         // LOG_LOADS_PER_TILE
      5,         // LOG_RAKING_THREADS,
      32,        // WART_GATHER_THRESHOLD,
      128 * 4,   // CTA_GATHER_THRESHOLD,
      7,         // LOG_SCHEDULE_GRANULARITY,
      gunrock::oprtr::advance::ALL_EDGES>
      EDGES_AdvanceKernelPolicy;

  typedef gunrock::oprtr::advance::KernelPolicy<
      Problem,   // Problem data type
      300,       // CUDA_ARCH
      8,         // MIN_CTA_OCCUPANCY,
      8,         // LOG_THREADS,
      8,         // LOG_BLOCKS,
      32 * 128,  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
      1,         // LOG_LOAD_VEC_SIZE,
      0,         // LOG_LOADS_PER_TILE
      5,         // LOG_RAKING_THREADS,
      32,        // WART_GATHER_THRESHOLD,
      128 * 4,   // CTA_GATHER_THRESHOLD,
      7,         // LOG_SCHEDULE_GRANULARITY,
      gunrock::oprtr::advance::LB_LIGHT>
      LB_LIGHT_AdvanceKernelPolicy;

  typedef gunrock::oprtr::advance::KernelPolicy<
      Problem,   // Problem data type
      300,       // CUDA_ARCH
      8,         // MIN_CTA_OCCUPANCY,
      7,         // LOG_THREADS,
      8,         // LOG_BLOCKS,
      32 * 128,  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
      1,         // LOG_LOAD_VEC_SIZE,
      0,         // LOG_LOADS_PER_TILE
      5,         // LOG_RAKING_THREADS,
      32,        // WART_GATHER_THRESHOLD,
      128 * 4,   // CTA_GATHER_THRESHOLD,
      7,         // LOG_SCHEDULE_GRANULARITY,
      gunrock::oprtr::advance::TWC_FORWARD>
      TWC_AdvanceKernelPolicy;

  typedef gunrock::oprtr::filter::KernelPolicy<Problem,  // Problem data type
                                               300,      // CUDA_ARCH
                                               0,        // SATURATION QUIT
                                               true,     // DEQUEUE_PROBLEM_SIZE
                                               4,        // MIN_CTA_OCCUPANCY
                                               9,        // LOG_THREADS
                                               1,        // LOG_LOAD_VEC_SIZE
                                               0,        // LOG_LOADS_PER_TILE
                                               5,        // LOG_RAKING_THREADS
                                               0,  // END_BITMASK (no bit-mask
                                                   // for cc)
                                               8,  // LOG_SCHEDULE_GRANULARITY
                                               gunrock::oprtr::filter::BY_PASS>
      FilterKernelPolicy;

  template <typename Dummy, gunrock::oprtr::advance::MODE A_MODE>
  struct MODE_SWITCH {};

  template <typename Dummy>
  struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB> {
    static cudaError_t Enact(Enactor &enactor) {
      return enactor.EnactCC<LB_AdvanceKernelPolicy, FilterKernelPolicy>();
    }
    static cudaError_t Init(Enactor &enactor, ContextPtr *context,
                            Problem *problem, int max_grid_size = 512) {
      return enactor.InitCC<LB_AdvanceKernelPolicy, FilterKernelPolicy>(
          context, problem, max_grid_size);
    }
  };

  template <typename Dummy>
  struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT> {
    static cudaError_t Enact(Enactor &enactor) {
      return enactor
          .EnactCC<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>();
    }
    static cudaError_t Init(Enactor &enactor, ContextPtr *context,
                            Problem *problem, int max_grid_size = 512) {
      return enactor.InitCC<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(
          context, problem, max_grid_size);
    }
  };

  template <typename Dummy>
  struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::TWC_FORWARD> {
    static cudaError_t Enact(Enactor &enactor) {
      return enactor.EnactCC<TWC_AdvanceKernelPolicy, FilterKernelPolicy>();
    }
    static cudaError_t Init(Enactor &enactor, ContextPtr *context,
                            Problem *problem, int max_grid_size = 512) {
      return enactor.InitCC<TWC_AdvanceKernelPolicy, FilterKernelPolicy>(
          context, problem, max_grid_size);
    }
  };

  template <typename Dummy>
  struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::ALL_EDGES> {
    static cudaError_t Enact(Enactor &enactor) {
      return enactor.EnactCC<EDGES_AdvanceKernelPolicy, FilterKernelPolicy>();
    }
    static cudaError_t Init(Enactor &enactor, ContextPtr *context,
                            Problem *problem, int max_grid_size = 512) {
      return enactor.InitCC<EDGES_AdvanceKernelPolicy, FilterKernelPolicy>(
          context, problem, max_grid_size);
    }
  };

  /**
   * @brief CC Enact kernel entry.
   *
   * @param[in] traversal_mode Mode of workload strategy in advance
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  // template <typename CCProblem>
  cudaError_t Enact(std::string traversal_mode = "LB") {
    if (this->min_sm_version >= 300) {
      if (traversal_mode == "LB")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>::Enact(*this);
      else if (traversal_mode == "LB_LIGHT")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>::Enact(
            *this);
      else if (traversal_mode == "TWC")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>::Enact(
            *this);
      else if (traversal_mode == "ALL_EDGES")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::ALL_EDGES>::Enact(
            *this);
    }

    // to reduce compile time, get rid of other architecture for now
    // TODO: add all the kernel policy settings for all archs
    printf("Not yet tuned for this architecture\n");
    return cudaErrorInvalidDeviceFunction;
  }

  /**
   * @brief CC Enact kernel entry.
   *
   * @param[in] context CudaContext pointer for ModernGPU API.
   * @param[in] problem Pointer to Problem object.
   * @param[in] traversal_mode Mode of workload strategy in advance
   * @param[in] max_grid_size Maximum grid size for kernel calls.
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  cudaError_t Init(ContextPtr *context, Problem *problem,
                   std::string traversal_mode = "LB", int max_grid_size = 512) {
    if (this->min_sm_version >= 300) {
      if (traversal_mode == "LB")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>::Init(
            *this, context, problem, max_grid_size);
      else if (traversal_mode == "LB_LIGHT")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>::Init(
            *this, context, problem, max_grid_size);
      else if (traversal_mode == "TWC")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>::Init(
            *this, context, problem, max_grid_size);
      else if (traversal_mode == "ALL_EDGES")
        return MODE_SWITCH<SizeT, gunrock::oprtr::advance::ALL_EDGES>::Init(
            *this, context, problem, max_grid_size);
    }

    // to reduce compile time, get rid of other architecture for now
    // TODO: add all the kernel policy settings for all archs
    printf("Not yet tuned for this architecture\n");
    return cudaErrorInvalidDeviceFunction;
  }
  /** @} */
};

}  // namespace cc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
