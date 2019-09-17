// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_iteration.cuh
 *
 * @brief IterationBase implementation
 */

#pragma once

#include <gunrock/graph/gp.cuh>
#include <gunrock/oprtr/advance/advance_base.cuh>
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/app/enactor_kernel.cuh>

namespace gunrock {
namespace app {

using IterationFlag = uint32_t;

enum : IterationFlag {
  Use_SubQ = 0x01,
  Use_FullQ = 0x02,
  Push = 0x10,
  Pull = 0x20,
  Update_Predecessors = 0x100,
  Unified_Receive = 0x200,
  Use_Double_Buffer = 0x400,
  Skip_Makeout_Selection = 0x800,
  Skip_PreScan = 0x1000,

  Iteration_Default = Use_FullQ | Push,
};

template <typename EnactorT, bool valid>
struct BoolSwitch {
  static cudaError_t UpdatePreds(EnactorT *enactor, int gpu_num,
                                 typename EnactorT::SizeT num_elements) {
    return cudaSuccess;
  }
};

template <typename EnactorT>
struct BoolSwitch<EnactorT, true> {
  static cudaError_t UpdatePreds(EnactorT *enactor, int gpu_num,
                                 typename EnactorT::SizeT num_elements) {
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT SizeT;
    cudaError_t retval = cudaSuccess;

    int k = gpu_num * enactor->num_gpus;
    if (enactor->flag & Size_Check) k += enactor->num_gpus;
    // int selector    = frontier_attribute->selector;
    int block_size = 256;
    int grid_size = num_elements / block_size;
    if ((num_elements % block_size) != 0) grid_size++;
    if (grid_size > 512) grid_size = 512;
    auto &enactor_slice = enactor->enactor_slices[k];
    auto &data_slice = enactor->problem->data_slices[gpu_num][0];
    auto &frontier = enactor_slice.frontier;
    auto &stream = enactor_slice.stream;
    auto &sub_graph = enactor->problem->sub_graphs[gpu_num];
    CopyPreds_Kernel<VertexT, SizeT><<<grid_size, block_size, 0, stream>>>(
        num_elements, frontier.V_Q()->GetPointer(util::DEVICE),
        data_slice.preds.GetPointer(util::DEVICE),
        data_slice.temp_preds.GetPointer(util::DEVICE));

    UpdatePreds_Kernel<VertexT, SizeT><<<grid_size, block_size, 0, stream>>>(
        num_elements, sub_graph.nodes, frontier.V_Q()->GetPointer(util::DEVICE),
        sub_graph.original_vertex.GetPointer(util::DEVICE),
        data_slice.temp_preds.GetPointer(util::DEVICE),
        data_slice.preds.GetPointer(util::DEVICE));

    return retval;
  }
};

/*
 * @brief IterationLoopBase data structure.
 * @tparam Iteration_Flag
 * @tparam Enactor
 */
template <typename _Enactor, IterationFlag _FLAG = Iteration_Default>
struct IterationLoopBase {
 public:
  typedef _Enactor Enactor;
  typedef typename Enactor::SizeT SizeT;
  typedef typename Enactor::ValueT ValueT;
  typedef typename Enactor::VertexT VertexT;
  typedef typename Enactor::Problem Problem;
  typedef typename Problem::DataSlice DataSlice;
  static const IterationFlag FLAG = _FLAG;

  Enactor *enactor;
  int gpu_num;
  IterationFlag flag;

  IterationLoopBase() : enactor(NULL), gpu_num(0) {}

  cudaError_t Init(Enactor *enactor, int gpu_num) {
    this->enactor = enactor;
    this->gpu_num = gpu_num;
    return cudaSuccess;
  }

  cudaError_t Core(int peer_) { return cudaSuccess; }

  cudaError_t Gather(int peer_) { return cudaSuccess; }

  bool Stop_Condition(int gpu_num = 0) { return All_Done(enactor[0], gpu_num); }

  cudaError_t Change() {
    auto &enactor_stats =
        enactor->enactor_slices[gpu_num * enactor->num_gpus].enactor_stats;
    enactor_stats.iteration++;
    return enactor_stats.retval;
  }

  cudaError_t UpdatePreds(SizeT num_elements) {
    cudaError_t retval = cudaSuccess;
    if (num_elements == 0) return retval;

    retval =
        BoolSwitch<Enactor, (FLAG & Update_Predecessors) != 0>::UpdatePreds(
            enactor, gpu_num, num_elements);
    return retval;
  }

  cudaError_t Check_Queue_Size(int peer_) {
    int k = gpu_num * enactor->num_gpus + peer_;
    auto &enactor_slice = enactor->enactor_slices[k];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;
    bool over_sized = false;
    // int  selector   = frontier_attribute->selector;
    auto iteration = enactor_stats.iteration;
    auto request_length = frontier.output_length[0] + 2;
    auto &retval = enactor_stats.retval;

    if (enactor->flag & Debug) {
      printf("%d\t %lld\t %d\t queue_size = %lld, output_length = %lld\n",
             gpu_num, (long long)iteration, peer_,
             (long long)(frontier.Next_V_Q()->GetSize()),
             (long long)request_length);
      fflush(stdout);
    }

    retval = CheckSize<SizeT, VertexT>(true, "queue3", request_length,
                                       frontier.Next_V_Q(), over_sized, gpu_num,
                                       iteration, peer_, false);
    if (retval) return retval;
    retval = CheckSize<SizeT, VertexT>(true, "queue3", request_length,
                                       frontier.V_Q(), over_sized, gpu_num,
                                       iteration, peer_, true);
    if (retval) return retval;
    // TODO
    // if (enactor -> problem -> use_double_buffer)
    //{
    //    if (enactor_stats->retval =
    //        Check_Size</*true,*/ SizeT, Value> (
    //            true, "queue3", request_length,
    //            &frontier_queue->values[selector^1], over_sized, thread_num,
    //            iteration, peer_, false)) return;
    //    if (enactor_stats->retval =
    //        Check_Size</*true,*/ SizeT, Value> (
    //            true, "queue3", request_length,
    //            &frontier_queue->values[selector  ], over_sized, thread_num,
    //            iteration, peer_, true )) return;
    //}
    return retval;
  }

  /*
   * @brief Make_Output function.
   * @tparam NUM_VERTEX_ASSOCIATES
   * @tparam NUM_VALUE__ASSOCIATES
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t MakeOutput(SizeT num_elements) {
    auto &mgpu_slice = enactor->mgpu_slices[gpu_num];
    int num_gpus = enactor->num_gpus;
    auto &enactor_slice =
        enactor->enactor_slices[gpu_num * num_gpus +
                                ((enactor->flag & Size_Check) ? 0 : num_gpus)];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &retval = enactor_stats.retval;
    auto &stream = enactor_slice.stream;
    auto &frontier = enactor_slice.frontier;
    auto &graph = enactor->problem->sub_graphs[gpu_num];

    if (num_gpus < 2) return retval;
    if (num_elements == 0) {
      for (int peer_ = 0; peer_ < num_gpus; peer_++) {
        mgpu_slice.out_length[peer_] = 0;
      }
      return retval;
    }
    bool over_sized = false, keys_over_sized = false;
    // int selector = frontier_attribute->selector;
    // printf("%d Make_Output begin, num_elements = %d, size_check = %s\n",
    //    data_slice -> gpu_idx, num_elements, enactor->size_check ? "true" :
    //    "false");
    // fflush(stdout);
    SizeT size_multi = 0;
    if (FLAG & Push) size_multi += 1;
    if (FLAG & Pull) size_multi += 1;

    for (int peer_ = 0; peer_ < num_gpus; peer_++) {
      if (retval = CheckSize<SizeT, VertexT>(
              enactor->flag & Size_Check, "keys_out", num_elements * size_multi,
              (peer_ == 0) ? enactor_slice.frontier.Next_V_Q()
                           : &(mgpu_slice.keys_out[peer_]),
              keys_over_sized, gpu_num, enactor_stats.iteration, peer_, false))
        break;
      // if (keys_over_sized)
      mgpu_slice.keys_outs[peer_] =
          (peer_ == 0)
              ? enactor_slice.frontier.Next_V_Q()->GetPointer(util::DEVICE)
              : mgpu_slice.keys_out[peer_].GetPointer(util::DEVICE);
      if (peer_ == 0) continue;

      over_sized = false;
      // for (i = 0; i< NUM_VERTEX_ASSOCIATES; i++)
      //{
      if (retval = CheckSize<SizeT, VertexT>(
              enactor->flag & Size_Check, "vertex_associate_outs",
              num_elements * NUM_VERTEX_ASSOCIATES * size_multi,
              &mgpu_slice.vertex_associate_out[peer_], over_sized, gpu_num,
              enactor_stats.iteration, peer_, false))
        break;
      // if (over_sized)
      mgpu_slice.vertex_associate_outs[peer_] =
          mgpu_slice.vertex_associate_out[peer_].GetPointer(util::DEVICE);
      //}
      // if (enactor_stats->retval) break;
      // if (over_sized)
      //    data_slice->vertex_associate_outs[peer_].Move(
      //        util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

      over_sized = false;
      // for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
      //{
      if (retval = CheckSize<SizeT, ValueT>(
              enactor->flag & Size_Check, "value__associate_outs",
              num_elements * NUM_VALUE__ASSOCIATES * size_multi,
              &mgpu_slice.value__associate_out[peer_], over_sized, gpu_num,
              enactor_stats.iteration, peer_, false))
        break;
      // if (over_sized)
      mgpu_slice.value__associate_outs[peer_] =
          mgpu_slice.value__associate_out[peer_].GetPointer(util::DEVICE);
      //}
      // if (enactor_stats->retval) break;
      // if (over_sized)
      //    data_slice->value__associate_outs[peer_].Move(
      //        util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
      if (FLAG & Skip_Makeout_Selection) break;
    }
    if (retval) return retval;

    if (FLAG & Skip_Makeout_Selection) {
      if (NUM_VALUE__ASSOCIATES == 0 && NUM_VERTEX_ASSOCIATES == 0) {
        // util::MemsetCopyVectorKernel<<<120, 512, 0, stream>>>(
        //    data_slice -> keys_out[1].GetPointer(util::DEVICE),
        //    frontier_queue -> keys[frontier_attribute ->
        //    selector].GetPointer(util::DEVICE), num_elements);
        mgpu_slice.keys_out[1].ForEach(
            frontier.V_Q()[0],
            [] __host__ __device__(VertexT & key_out, const VertexT &key_in) {
              key_out = key_in;
            },
            num_elements, util::DEVICE, stream);
        for (int peer_ = 0; peer_ < num_gpus; peer_++)
          mgpu_slice.out_length[peer_] = num_elements;
        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize failed", __FILE__,
                                   __LINE__))
          return retval;
        return retval;
      } else {
        for (int peer_ = 2; peer_ < num_gpus; peer_++) {
          mgpu_slice.keys_out[peer_].SetPointer(
              mgpu_slice.keys_out[1].GetPointer(util::DEVICE),
              mgpu_slice.keys_out[1].GetSize(), util::DEVICE);
          mgpu_slice.keys_outs[peer_] =
              mgpu_slice.keys_out[peer_].GetPointer(util::DEVICE);

          mgpu_slice.vertex_associate_out[peer_].SetPointer(
              mgpu_slice.vertex_associate_out[1].GetPointer(util::DEVICE),
              mgpu_slice.vertex_associate_out[1].GetSize(), util::DEVICE);
          mgpu_slice.vertex_associate_outs[peer_] =
              mgpu_slice.vertex_associate_out[peer_].GetPointer(util::DEVICE);

          mgpu_slice.value__associate_out[peer_].SetPointer(
              mgpu_slice.value__associate_out[1].GetPointer(util::DEVICE),
              mgpu_slice.value__associate_out[1].GetSize(), util::DEVICE);
          mgpu_slice.value__associate_outs[peer_] =
              mgpu_slice.value__associate_out[peer_].GetPointer(util::DEVICE);
        }
      }
    }
    // printf("%d Make_Out 1\n", data_slice -> gpu_idx);
    // fflush(stdout);
    // if (keys_over_sized)
    mgpu_slice.keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
    mgpu_slice.vertex_associate_outs.Move(util::HOST, util::DEVICE, num_gpus, 0,
                                          stream);
    mgpu_slice.value__associate_outs.Move(util::HOST, util::DEVICE, num_gpus, 0,
                                          stream);
    // util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PreMakeOut",
    //    frontier_queue -> keys[frontier_attribute ->
    //    selector].GetPointer(util::DEVICE), num_elements, data_slice ->
    //    gpu_idx, enactor_stats -> iteration, -1, stream);
    int block_size = 512;
    int grid_size = num_elements / block_size / 2 + 1;
    if (grid_size > 480) grid_size = 480;
    // printf("%d Make_Out 2, num_blocks = %d, num_threads = %d\n",
    //    data_slice -> gpu_idx, num_blocks, AdvanceKernelPolicy::THREADS);
    // fflush(stdout);
    if ((FLAG & Skip_Makeout_Selection) == 0) {
      for (int i = 0; i < num_gpus; i++) mgpu_slice.out_length[i] = 1;
      mgpu_slice.out_length.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
      // printf("Make_Output direction = %s %s\n", FORWARD ? "FORWARD" : "",
      // BACKWARD ? "BACKWARD" : "");

      /*printf("num_blocks = %d, num_threads = %d, stream = %p, "
          "num_elements = %d, num_gpus = %d, out_length = %p, (%d)"
          "keys_in = %p (%d), partition_table = %p (%d), convertion_table = %d
         (%d), " "vertex_associate_orgs = %p (%d), value__associate_orgs = %p
         (%d), " "keys_outs = %p (%d), vertex_associate_outs = %p (%d),
         value__associate_outs = %p (%d), " "keep_node_num = %s,
         num_vertex_associates = %d, num_value_associates = %d\n", num_blocks,
         AdvanceKernelPolicy::THREADS /2, stream, num_elements, num_gpus,
          data_slice -> out_length.GetPointer(util::DEVICE), data_slice ->
         out_length.GetSize(), frontier_queue -> keys[frontier_attribute ->
         selector].GetPointer(util::DEVICE), frontier_queue ->
         keys[frontier_attribute -> selector].GetSize(), graph_slice ->
         partition_table      .GetPointer(util::DEVICE), graph_slice ->
         partition_table      .GetSize(), graph_slice -> convertion_table
         .GetPointer(util::DEVICE), graph_slice -> convertion_table .GetSize(),
          data_slice  -> vertex_associate_orgs[0],
          data_slice  -> vertex_associate_orgs.GetSize(),
          data_slice  -> value__associate_orgs[0],
          data_slice  -> value__associate_orgs.GetSize(),
          data_slice  -> keys_outs            .GetPointer(util::DEVICE),
          data_slice  -> keys_outs            .GetSize(),
          data_slice  -> vertex_associate_outs[1],
          data_slice  -> vertex_associate_outs.GetSize(),
          data_slice  -> value__associate_outs[1],
          data_slice  -> value__associate_outs.GetSize(),
          enactor -> problem -> keep_node_num ? "true" : "false",
          NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES);*/

      if (FLAG & Push)
        MakeOutput_Kernel<VertexT, SizeT, ValueT, NUM_VERTEX_ASSOCIATES,
                          NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, 0, stream>>>(
                num_elements, num_gpus,
                mgpu_slice.out_length.GetPointer(util::DEVICE),
                frontier.V_Q()->GetPointer(util::DEVICE),
                graph.partition_table.GetPointer(util::DEVICE),
                graph.convertion_table.GetPointer(util::DEVICE),
                mgpu_slice.vertex_associate_orgs.GetPointer(util::DEVICE),
                mgpu_slice.value__associate_orgs.GetPointer(util::DEVICE),
                mgpu_slice.keys_outs.GetPointer(util::DEVICE),
                mgpu_slice.vertex_associate_outs.GetPointer(util::DEVICE),
                mgpu_slice.value__associate_outs.GetPointer(util::DEVICE),
                enactor->problem->flag & partitioner::Keep_Node_Num);

      if (FLAG & Pull)
        MakeOutput_Backward_Kernel<VertexT, SizeT, ValueT,
                                   NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, 0, stream>>>(
                num_elements, num_gpus,
                mgpu_slice.out_length.GetPointer(util::DEVICE),
                frontier.V_Q()->GetPointer(util::DEVICE),
                graph.backward_offset.GetPointer(util::DEVICE),
                graph.backward_partition.GetPointer(util::DEVICE),
                graph.backward_convertion.GetPointer(util::DEVICE),
                mgpu_slice.vertex_associate_orgs.GetPointer(util::DEVICE),
                mgpu_slice.value__associate_orgs.GetPointer(util::DEVICE),
                mgpu_slice.keys_outs.GetPointer(util::DEVICE),
                mgpu_slice.vertex_associate_outs.GetPointer(util::DEVICE),
                mgpu_slice.value__associate_outs.GetPointer(util::DEVICE),
                enactor->problem->flag & partitioner::Keep_Node_Num);

      mgpu_slice.out_length.Move(util::DEVICE, util::HOST, num_gpus, 0, stream);
      frontier.queue_index++;
    } else {
      MakeOutput_SkipSelection_Kernel<
          VertexT, SizeT, ValueT, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
          <<<grid_size, block_size, 0, stream>>>(
              num_elements, frontier.V_Q()->GetPointer(util::DEVICE),
              mgpu_slice.vertex_associate_orgs.GetPointer(util::DEVICE),
              mgpu_slice.value__associate_orgs.GetPointer(util::DEVICE),
              mgpu_slice.keys_out[1].GetPointer(util::DEVICE),
              mgpu_slice.vertex_associate_out[1].GetPointer(util::DEVICE),
              mgpu_slice.value__associate_out[1].GetPointer(util::DEVICE));
      for (int peer_ = 0; peer_ < num_gpus; peer_++)
        mgpu_slice.out_length[peer_] = num_elements;
    }
    if (retval = util::GRError(cudaStreamSynchronize(stream),
                               "Make_Output failed", __FILE__, __LINE__))
      return retval;
    if ((FLAG & Skip_Makeout_Selection) == 0) {
      for (int i = 0; i < num_gpus; i++) {
        mgpu_slice.out_length[i]--;
        // printf("out_length[%d] = %d\n", i, data_slice -> out_length[i]);
      }
    }
    // for (int i=0; i<num_gpus; i++)
    //{
    // if (i == 0)
    //    printf("%d, selector = %d, keys = %p\n",
    //        data_slice -> gpu_idx, frontier_attribute -> selector^1,
    //        data_slice -> keys_outs[i]);
    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PostMakeOut",
    //        data_slice -> keys_outs[i], data_slice -> out_length[i],
    //        data_slice -> gpu_idx, enactor_stats -> iteration, i, stream);
    //}

    // printf("%d Make_Out 3\n", data_slice -> gpu_idx);
    // fflush(stdout);

    return retval;
  }

  /*template <
      int NUM_VERTEX_ASSOCIATES,
      int NUM_VALUE__ASSOCIATES>
  cudaError_t Expand_Incoming(int peer_)
  {
      return cudaSuccess;
  }*/

  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES,
            typename ExpandOpT>
  cudaError_t ExpandIncomingBase(SizeT &received_length, int peer_,
                                 ExpandOpT exapnd_op) {
    bool over_sized = false;
    auto &mgpu_slice = enactor->mgpu_slices[gpu_num];
    auto &enactor_slice =
        enactor->enactor_slices[gpu_num * enactor->num_gpus +
                                ((FLAG & Unified_Receive) ? 0 : peer_)];
    auto &iteration = enactor_slice.enactor_stats.iteration;
    auto &out_length = mgpu_slice.in_length_out;
    auto &num_elements = mgpu_slice.in_length[iteration % 2][peer_];
    auto &keys_in = mgpu_slice.keys_in[iteration % 2][peer_];
    auto &vertex_associate_in =
        mgpu_slice.vertex_associate_in[iteration % 2][peer_];
    auto &value__associate_in =
        mgpu_slice.value__associate_in[iteration % 2][peer_];
    auto &retval = enactor_slice.enactor_stats.retval;
    auto &frontier = enactor_slice.frontier;
    auto &stream = enactor_slice.stream;

    if (FLAG & Unified_Receive) {
      retval = CheckSize<SizeT, VertexT>(
          enactor->flag & Size_Check, "incoming_queue",
          num_elements + received_length, frontier.V_Q(), over_sized, gpu_num,
          iteration, peer_, true);
      if (retval) return retval;
      received_length += num_elements;
    } else {
      retval = CheckSize<SizeT, VertexT>(
          enactor->flag & Size_Check, "incomping_queue", num_elements,
          frontier.V_Q(), over_sized, gpu_num, iteration, peer_, false);
      if (retval) return retval;
      out_length[peer_] = 0;
      GUARD_CU(out_length.Move(util::HOST, util::DEVICE, 1, peer_, stream));
    }

    int block_size = 512;
    int grid_size = num_elements / block_size + 1;
    if (grid_size > 240) grid_size = 240;
    ExpandIncoming_Kernel<VertexT, SizeT, ValueT, NUM_VERTEX_ASSOCIATES,
                          NUM_VALUE__ASSOCIATES>
        <<<grid_size, block_size, 0, stream>>>(
            gpu_num, num_elements, keys_in.GetPointer(util::DEVICE),
            vertex_associate_in.GetPointer(util::DEVICE),
            value__associate_in.GetPointer(util::DEVICE),
            out_length.GetPointer(util::DEVICE) +
                ((FLAG & Unified_Receive) ? 0 : peer_),
            frontier.V_Q()->GetPointer(util::DEVICE), exapnd_op);

    GUARD_CU(out_length.Move(util::DEVICE, util::HOST, 1,
                             (FLAG & Unified_Receive) ? 0 : peer_, stream));
    return retval;
  }

  cudaError_t Compute_OutputLength(int peer_) {
    cudaError_t retval = cudaSuccess;
    bool over_sized = false;
    auto &enactor_slice =
        enactor->enactor_slices[gpu_num * enactor->num_gpus + peer_];
    auto &frontier = enactor_slice.frontier;
    auto &stream = enactor_slice.stream;
    auto &graph = enactor->problem->sub_graphs[gpu_num];

    if ((enactor->flag & Size_Check) == 0 && (flag & Skip_PreScan)) {
      frontier.output_length[0] = 0;
      return retval;
    }

    retval = CheckSize<SizeT, SizeT>(
        enactor->flag & Size_Check, "scanned_edges", frontier.queue_length + 2,
        &frontier.output_offsets, over_sized, -1, -1, -1, false);
    if (retval) return retval;

    GUARD_CU(oprtr::ComputeOutputLength<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), enactor_slice.oprtr_parameters));

    GUARD_CU(
        frontier.output_length.Move(util::DEVICE, util::HOST, 1, 0, stream));
    return retval;
  }
};

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
