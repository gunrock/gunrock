// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_helper.cuh
 *
 * @brief helper functions for enactor base
 */

#pragma once

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/*
 * @brief Check whether there is any error or all work load done
 * @tparam EnactorT
 * @param[in] enactor pointer to enactor
 * @param[in] gpu_num
 */
template <typename EnactorT>
bool All_Done(EnactorT &enactor, int gpu_num = 0) {
  int num_gpus = enactor.num_gpus;
  auto &enactor_slices = enactor.enactor_slices;

  for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
    auto &retval = enactor_slices[gpu].enactor_stats.retval;
    if (retval == cudaSuccess) continue;
    printf("(CUDA error %d @ GPU %d: %s\n", retval, gpu % num_gpus,
           cudaGetErrorString(retval));
    fflush(stdout);
    return true;
  }

  // if (enactor_slices[0].enactor_stats.iteration > 2)
  //    return true;

  for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
    auto &frontier = enactor_slices[gpu].frontier;
    if (frontier.queue_length != 0)  // || frontier.has_incoming)
    {
      // printf("frontier_attribute[%d].queue_length = %d\n",
      //    gpu,frontier_attribute[gpu].queue_length);
      return false;
    }
  }

  if (num_gpus == 1) return true;

  auto &mgpu_slices = enactor.mgpu_slices;
  for (int gpu = 0; gpu < num_gpus; gpu++)
    for (int peer = 1; peer < num_gpus; peer++)
      for (int i = 0; i < 2; i++)
        if (mgpu_slices[gpu].in_length[i][peer] != 0) {
          // printf("data_slice[%d]->in_length[%d][%d] = %d\n",
          //    gpu, i, peer, data_slice[gpu]->in_length[i][peer]);
          return false;
        }

  for (int gpu = 0; gpu < num_gpus; gpu++)
    for (int peer = 1; peer < num_gpus; peer++)
      if (mgpu_slices[gpu].out_length[peer] != 0) {
        // printf("data_slice[%d]->out_length[%d] = %d\n",
        //    gpu, peer, data_slice[gpu]->out_length[peer]);
        return false;
      }

  return true;
}

/*
 * @brief Check size function.
 *
 * @tparam SIZE_CHECK
 * @tparam SizeT
 * @tparam Type
 *
 * @param[in] name
 * @param[in] target_length
 * @param[in] array
 * @param[in] oversized
 * @param[in] thread_num
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] keep_content
 *
 * \return cudaError_t object Indicates the success of all CUDA calls.
 */
template <
    // bool     SIZE_CHECK,
    typename SizeT, typename Type>
cudaError_t CheckSize(bool size_check, const char *name, SizeT target_length,
                      util::Array1D<SizeT, Type> *array, bool &oversized,
                      int thread_num = -1, int iteration = -1, int peer_ = -1,
                      bool keep_content = false) {
  cudaError_t retval = cudaSuccess;

  if (array == NULL) return retval;

  if (target_length > array->GetSize()) {
    printf("%d\t %d\t %d\t %s \t oversize :\t %lld ->\t %lld\n", thread_num,
           iteration, peer_, name, (long long)array->GetSize(),
           (long long)target_length);
    // fflush(stdout);
    oversized = true;
    if (size_check) {
      if (array->GetSize() != 0)
        retval = array->EnsureSize(target_length, keep_content);
      else
        retval = array->Allocate(target_length, util::DEVICE);
    } else {
      char str[256];
      // memcpy(str, name, sizeof(char) * strlen(name));
      // memcpy(str + strlen(name), temp_str, sizeof(char) * strlen(temp_str));
      // str[strlen(name)+strlen(temp_str)]='0';
      sprintf(str, "%s oversized", name);
      retval =
          util::GRError(cudaErrorLaunchOutOfResources, str, __FILE__, __LINE__);
    }
  }
  return retval;
}

/*
 * @brief Check size function.
 *
 * @tparam SIZE_CHECK
 * @tparam SizeT
 * @tparam VertexId
 * @tparam Value
 * @tparam GraphSlice
 * @tparam DataSlice
 * @tparam num_vertex_associate
 * @tparam num_value__associate
 *
 * @param[in] gpu
 * @param[in] peer
 * @param[in] array
 * @param[in] queue_length
 * @param[in] enactor_stats
 * @param[in] data_slice_l
 * @param[in] data_slice_p
 * @param[in] graph_slice_l Graph slice local
 * @param[in] graph_slice_p
 * @param[in] stream CUDA stream.
 */
template <
    // bool     SIZE_CHECK,
    // typename VertexId,
    // typename SizeT,
    // typename Value,
    typename Enactor, typename GraphSliceT, typename DataSlice,
    int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
void PushNeighbor_Old(Enactor *enactor, int gpu, int peer,
                      typename Enactor::SizeT queue_length,
                      EnactorStats<typename Enactor::SizeT> *enactor_stats,
                      DataSlice *data_slice_l, DataSlice *data_slice_p,
                      GraphSliceT *graph_slice_l, GraphSliceT *graph_slice_p,
                      cudaStream_t stream) {
  typedef typename Enactor::VertexId VertexId;
  typedef typename Enactor::SizeT SizeT;
  typedef typename Enactor::Value Value;

  if (peer == gpu) return;
  int gpu_ = peer < gpu ? gpu : gpu + 1;
  int peer_ = peer < gpu ? peer + 1 : peer;
  int i, t = enactor_stats->iteration % 2;
  bool to_reallocate = false;
  bool over_sized = false;

  data_slice_p->in_length[enactor_stats->iteration % 2][gpu_] = queue_length;
  if (queue_length == 0) return;

  if (data_slice_p->keys_in[t][gpu_].GetSize() < queue_length)
    to_reallocate = true;
  else {
    for (i = 0; i < NUM_VERTEX_ASSOCIATES; i++) {
      if (data_slice_p->vertex_associate_in[t][gpu_][i].GetSize() <
          queue_length) {
        to_reallocate = true;
        break;
      }
    }
    if (!to_reallocate) {
      for (i = 0; i < NUM_VALUE__ASSOCIATES; i++) {
        if (data_slice_p->value__associate_in[t][gpu_][i].GetSize() <
            queue_length) {
          to_reallocate = true;
          break;
        }
      }
    }
  }

  if (to_reallocate) {
    if (enactor->size_check) util::SetDevice(data_slice_p->gpu_idx);
    if (enactor_stats->retval = CheckSize<SizeT, VertexId>(
            enactor->size_check, "keys_in", queue_length,
            &data_slice_p->keys_in[t][gpu_], over_sized, gpu,
            enactor_stats->iteration, peer))
      return;

    for (i = 0; i < NUM_VERTEX_ASSOCIATES; i++) {
      if (enactor_stats->retval = CheckSize<SizeT, VertexId>(
              enactor->size_check, "vertex_associate_in", queue_length,
              &data_slice_p->vertex_associate_in[t][gpu_][i], over_sized, gpu,
              enactor_stats->iteration, peer))
        return;
      data_slice_p->vertex_associate_ins[t][gpu_][i] =
          data_slice_p->vertex_associate_in[t][gpu_][i].GetPointer(
              util::DEVICE);
    }
    for (i = 0; i < NUM_VALUE__ASSOCIATES; i++) {
      if (enactor_stats->retval = CheckSize<SizeT, Value>(
              enactor->size_check, "value__associate_in", queue_length,
              &data_slice_p->value__associate_in[t][gpu_][i], over_sized, gpu,
              enactor_stats->iteration, peer))
        return;
      data_slice_p->value__associate_ins[t][gpu_][i] =
          data_slice_p->value__associate_in[t][gpu_][i].GetPointer(
              util::DEVICE);
    }
    if (enactor->size_check) {
      if (enactor_stats->retval =
              data_slice_p->vertex_associate_ins[t][gpu_].Move(util::HOST,
                                                               util::DEVICE))
        return;
      if (enactor_stats->retval =
              data_slice_p->value__associate_ins[t][gpu_].Move(util::HOST,
                                                               util::DEVICE))
        return;
      util::SetDevice(data_slice_l->gpu_idx);
    }
  }

  if (enactor_stats->retval = util::GRError(
          cudaMemcpyAsync(
              data_slice_p->keys_in[t][gpu_].GetPointer(util::DEVICE),
              data_slice_l->keys_out[peer_].GetPointer(util::DEVICE),
              sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
          "cudaMemcpyPeer keys failed", __FILE__, __LINE__))
    return;

  for (int i = 0; i < NUM_VERTEX_ASSOCIATES; i++) {
    if (enactor_stats->retval = util::GRError(
            cudaMemcpyAsync(data_slice_p->vertex_associate_ins[t][gpu_][i],
                            data_slice_l->vertex_associate_outs[peer_][i],
                            sizeof(VertexId) * queue_length, cudaMemcpyDefault,
                            stream),
            "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__))
      return;
  }

  for (int i = 0; i < NUM_VALUE__ASSOCIATES; i++) {
    if (enactor_stats->retval = util::GRError(
            cudaMemcpyAsync(data_slice_p->value__associate_ins[t][gpu_][i],
                            data_slice_l->value__associate_outs[peer_][i],
                            sizeof(Value) * queue_length, cudaMemcpyDefault,
                            stream),
            "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__))
      return;
  }
}

template <typename Enactor, int NUM_VERTEX_ASSOCIATES,
          int NUM_VALUE__ASSOCIATES>
cudaError_t PushNeighbor(Enactor &enactor, int gpu, int peer) {
  typedef typename Enactor::VertexT VertexT;
  typedef typename Enactor::SizeT SizeT;
  typedef typename Enactor::ValueT ValueT;

  if (peer == gpu) return cudaSuccess;
  int gpu_ = peer < gpu ? gpu : gpu + 1;
  int peer_ = peer < gpu ? peer + 1 : peer;
  auto &enactor_slice = enactor.enactor_slices[gpu * enactor.num_gpus + peer_];
  auto &enactor_stats = enactor_slice.enactor_stats;
  cudaError_t retval = enactor_stats.retval;
  auto &mgpu_slice_l = enactor.mgpu_slices[gpu];
  auto &mgpu_slice_p = enactor.mgpu_slices[peer];
  auto queue_length = mgpu_slice_l.out_length[peer_];
  int t = enactor_stats.iteration % 2;
  bool to_reallocate = false;
  bool over_sized = false;

  mgpu_slice_p.in_length[t][gpu_] = queue_length;
  mgpu_slice_p.in_iteration[t][gpu_] = enactor_stats.iteration;
  if (queue_length == 0) return retval;

  if (enactor.communicate_multipy > 1)
    queue_length *= enactor.communicate_multipy;

  if (mgpu_slice_l.keys_out[peer_].GetPointer(util::DEVICE) != NULL &&
      mgpu_slice_p.keys_in[t][gpu_].GetSize() < queue_length)
    to_reallocate = true;
  if (mgpu_slice_p.vertex_associate_in[t][gpu_].GetSize() <
      queue_length * NUM_VERTEX_ASSOCIATES)
    to_reallocate = true;
  if (mgpu_slice_p.value__associate_in[t][gpu_].GetSize() <
      queue_length * NUM_VALUE__ASSOCIATES)
    to_reallocate = true;

  if (to_reallocate) {
    if (enactor.flag & Size_Check) {
      GUARD_CU(util::SetDevice(mgpu_slice_p.gpu_idx));
    }
    if (mgpu_slice_l.keys_out[peer_].GetPointer(util::DEVICE) != NULL) {
      retval = CheckSize<SizeT, VertexT>(
          enactor.flag & Size_Check, "keys_in", queue_length,
          &mgpu_slice_p.keys_in[t][gpu_], over_sized, gpu,
          enactor_stats.iteration, peer);
      if (retval) return retval;
    }
    retval = CheckSize<SizeT, VertexT>(
        enactor.flag & Size_Check, "vertex_associate_in",
        queue_length * NUM_VERTEX_ASSOCIATES,
        &mgpu_slice_p.vertex_associate_in[t][gpu_], over_sized, gpu,
        enactor_stats.iteration, peer);
    if (retval) return retval;
    retval = CheckSize<SizeT, ValueT>(
        enactor.flag & Size_Check, "value__associate_in",
        queue_length * NUM_VALUE__ASSOCIATES,
        &mgpu_slice_p.value__associate_in[t][gpu_], over_sized, gpu,
        enactor_stats.iteration, peer);
    if (retval) return retval;
    if (enactor.flag & Size_Check) {
      GUARD_CU(util::SetDevice(mgpu_slice_l.gpu_idx));
    }
  }

  auto &stream = enactor_slice.stream2;
  if (mgpu_slice_l.keys_out[peer_].GetPointer(util::DEVICE) != NULL) {
    // util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys_out",
    //    data_slice_l -> keys_out[peer_].GetPointer(util::DEVICE),
    //    queue_length, gpu, enactor_stats -> iteration, peer_, stream);

    GUARD_CU2(cudaMemcpyAsync(
                  mgpu_slice_p.keys_in[t][gpu_].GetPointer(util::DEVICE),
                  mgpu_slice_l.keys_out[peer_].GetPointer(util::DEVICE),
                  sizeof(VertexT) * queue_length, cudaMemcpyDefault, stream),
              "cudamemcpyPeer keys failed");

    // printf("%d @ %p -> %d @ %p, size = %d\n",
    //    gpu , data_slice_l -> keys_out[peer_].GetPointer(util::DEVICE),
    //    peer, data_slice_p -> keys_in[t][gpu_].GetPointer(util::DEVICE),
    //    sizeof(VertexId) * queue_length);
  } else {
    // printf("push key skiped\n");
  }
  if (NUM_VERTEX_ASSOCIATES != 0) {
    GUARD_CU2(
        cudaMemcpyAsync(
            mgpu_slice_p.vertex_associate_in[t][gpu_].GetPointer(util::DEVICE),
            mgpu_slice_l.vertex_associate_out[peer_].GetPointer(util::DEVICE),
            sizeof(VertexT) * queue_length * NUM_VERTEX_ASSOCIATES,
            cudaMemcpyDefault, stream),
        "cudamemcpyPeer keys failed");
  }
  if (NUM_VALUE__ASSOCIATES != 0) {
    GUARD_CU2(
        cudaMemcpyAsync(
            mgpu_slice_p.value__associate_in[t][gpu_].GetPointer(util::DEVICE),
            mgpu_slice_l.value__associate_out[peer_].GetPointer(util::DEVICE),
            sizeof(ValueT) * queue_length * NUM_VALUE__ASSOCIATES,
            cudaMemcpyDefault, stream),
        "cudamemcpyPeer keys failed");
  }

#ifdef ENABLE_PERFORMANCE_PROFILING
  // enactor_stats -> iter_out_length.back().push_back(queue_length);
#endif
  return retval;
}

/*
 * @brief Show debug information
 * @tparam EnactorT
 * @param[in] enactor pointer to enactor
 * @param[in] gpu_num
 * @param[in] peer_
 * @param[in] check_name
 * @param[in] stream CUDA stream.
 */
template <typename EnactorT>
void ShowDebugInfo(EnactorT &enactor, int gpu_num, int peer_,
                   std::string check_name = "", cudaStream_t stream = 0) {
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::ValueT ValueT;

  auto &enactor_slice =
      enactor.enactor_slices[gpu_num * enactor.num_gpus + peer_];
  // auto &data_slice = enactor.problem -> data_slices[gpu_num];
  auto &mgpu_slice = enactor.mgpu_slices[gpu_num];
  auto queue_length = enactor_slice.frontier.queue_length;

  // util::cpu_mt::PrintMessage(check_name.c_str(), thread_num,
  // enactor_stats->iteration); printf("%d \t %d\t \t reset = %d, index =
  // %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset,
  // frontier_attribute->queue_index);fflush(stdout); if
  // (frontier_attribute->queue_reset)
  //    queue_length = frontier_attribute->queue_length;
  // else if (enactor_stats->retval =
  // util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index,
  // queue_length, false, stream), "work_progress failed", __FILE__, __LINE__))
  // return; util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+"
  // Queue_Length").c_str(), &(queue_length), 1, thread_num,
  // enactor_stats->iteration);
  printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %lld\n", gpu_num,
         enactor_slice.enactor_stats.iteration, peer_, mgpu_slice.stages[peer_],
         check_name.c_str(), (long long)queue_length);
  fflush(stdout);
  // printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p =
  // %p\n",thread_num, enactor_stats->iteration, peer_,
  // frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
  // util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(),
  // data_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE),
  // queue_length, thread_num, enactor_stats->iteration,peer_, stream); if
  // (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
  //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1",
  //    graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE),
  //    _queue_length, thread_num, enactor_stats->iteration);
  // util::cpu_mt::PrintGPUArray<SizeT, VertexId>("degrees",
  // data_slice->degrees.GetPointer(util::DEVICE), graph_slice->nodes,
  // thread_num, enactor_stats->iteration); if (BFSProblem::MARK_PREDECESSOR)
  //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1",
  //    data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes,
  //    thread_num, enactor_stats->iteration);
  // if (BFSProblem::ENABLE_IDEMPOTENCE)
  //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1",
  //    data_slice[0]->visited_mask.GetPointer(util::DEVICE),
  //    (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
}

/*
 * @brief Set record function.
 *
 * @tparam DataSlice
 *
 * @param[in] data_slice
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] stage
 * @param[in] stream CUDA stream.
 */
template <typename MgpuSliceT>
cudaError_t SetRecord(MgpuSliceT &mgpu_slice, int iteration, int peer_,
                      int stage, cudaStream_t stream) {
  cudaError_t retval =
      cudaEventRecord(mgpu_slice.events[iteration % 4][peer_][stage], stream);
  mgpu_slice.events_set[iteration % 4][peer_][stage] = true;
  return retval;
}

/*
 * @brief Set record function.
 *
 * @tparam DataSlice
 *
 * @param[in] data_slice
 * @param[in] iteration
 * @param[in] peer_
 * @param[in] stage_to_check
 * @param[in] stage
 * @param[in] to_show
 */
template <typename MgpuSliceT>
cudaError_t CheckRecord(MgpuSliceT &mgpu_slice, int iteration, int peer_,
                        int stage_to_check, int &stage, bool &to_show) {
  cudaError_t retval = cudaSuccess;
  to_show = true;
  if (!mgpu_slice.events_set[iteration % 4][peer_][stage_to_check]) {
    to_show = false;
  } else {
    retval =
        cudaEventQuery(mgpu_slice.events[iteration % 4][peer_][stage_to_check]);
    if (retval == cudaErrorNotReady) {
      to_show = false;
      retval = cudaSuccess;
    } else if (retval == cudaSuccess) {
      mgpu_slice.events_set[iteration % 4][peer_][stage_to_check] = false;
    }
  }
  return retval;
}

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
