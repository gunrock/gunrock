// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/track_utils.cuh>
#include <gunrock/app/bfs/bfs_kernel.cuh>

namespace gunrock {
namespace app {
namespace bfs {

/**
 * @brief Speciflying parameters for BFS Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<bool>(
      "idempotence",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to enable idempotence optimization", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "direction-optimized",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to enable direction optimizing BFS", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "do-a",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0.001, "Threshold to switch from forward-push to backward-pull in DOBFS",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "do-b",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0.200, "Threshold to switch from backward-pull to forward-push in DOBFS",
      __FILE__, __LINE__));

  return retval;
}

/**
 * @brief defination of BFS iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct BFSIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push |
                                             (((EnactorT::Problem::FLAG &
                                                Mark_Predecessors) != 0)
                                                  ? Update_Predecessors
                                                  : 0x0)> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem ProblemT;
  typedef typename ProblemT::GraphT::CsrT CsrT;
  typedef typename ProblemT::GraphT::GpT GpT;
  typedef typename ProblemT::MaskT MaskT;
  typedef typename ProblemT::LabelT LabelT;

  typedef IterationLoopBase<EnactorT,
                            Use_FullQ | Push |
                                (((ProblemT::FLAG & Mark_Predecessors) != 0)
                                     ? Update_Predecessors
                                     : 0x0)>
      BaseIterationLoop;

  BFSIterationLoop() : BaseIterationLoop() {}

  cudaError_t Gather(int peer_) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &graph = data_slice.sub_graph[0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;

    data_slice.num_visited_vertices += frontier.queue_length;
    data_slice.num_unvisited_vertices =
        graph.nodes - data_slice.num_visited_vertices;
    float predicted_backward_visits =
        (data_slice.num_visited_vertices == 0)
            ? (std::numeric_limits<float>::infinity())
            : (data_slice.num_unvisited_vertices * 1.0 * graph.nodes /
               data_slice.num_visited_vertices);
    float predicted_forward_visits =
        frontier.queue_length * 1.0 * graph.edges / graph.nodes;

    auto iteration_ = enactor_stats.iteration % 4;
    if (this->enactor->direction_optimized) {
      if (data_slice.previous_direction == FORWARD) {
        if (predicted_forward_visits >
                predicted_backward_visits * this->enactor->do_a &&
            !data_slice.been_in_backward)
          data_slice.direction_votes[iteration_] = BACKWARD;
        else
          data_slice.direction_votes[iteration_] = FORWARD;
      } else {
        data_slice.been_in_backward = true;
        if (predicted_forward_visits >
            predicted_backward_visits * this->enactor->do_b)
          data_slice.direction_votes[iteration_] = BACKWARD;
        else
          data_slice.direction_votes[iteration_] = FORWARD;
      }
    } else
      data_slice.direction_votes[iteration_] = FORWARD;
    data_slice.direction_votes[(iteration_ + 1) % 4] = UNDECIDED;

    if (this->enactor->num_gpus > 1 && enactor_stats.iteration != 0 &&
        this->enactor->direction_optimized) {
      while (
          this->enactor->problem->data_slices[0]->direction_votes[iteration_] ==
          UNDECIDED) {
        sleep(0);
      }
      data_slice.current_direction =
          this->enactor->problem->data_slices[0]->direction_votes[iteration_];
    } else if (enactor_stats.iteration == 0)
      data_slice.direction_votes[iteration_] = FORWARD;
    else {
      data_slice.current_direction = data_slice.direction_votes[iteration_];
    }

    return retval;
  }

  cudaError_t Compute_OutputLength(int peer_) {
    cudaError_t retval = cudaSuccess;
    bool over_sized = false;
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &frontier = enactor_slice.frontier;
    auto &stream = enactor_slice.stream;
    auto &graph = this->enactor->problem->sub_graphs[this->gpu_num];

    if (((this->enactor->flag & Size_Check) == 0 &&
         ((this->flag & Skip_PreScan) != 0)) ||
        (this->enactor->problem->data_slices[0]->current_direction == BACKWARD))
    //(AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_FORWARD ||
    // AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_BACKWARD))
    {
      frontier.output_length[0] = 0;
    }

    else {
      // printf("Size check runs\n");
      retval = CheckSize<SizeT, SizeT>(
          this->enactor->flag & Size_Check, "scanned_edges",
          frontier.queue_length + 2, &frontier.output_offsets, over_sized, -1,
          -1, -1, false);
      if (retval) return retval;

      GUARD_CU(oprtr::ComputeOutputLength<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), enactor_slice.oprtr_parameters));

      GUARD_CU(
          frontier.output_length.Move(util::DEVICE, util::HOST, 1, 0, stream));
    }
    return retval;
  }

  cudaError_t Check_Queue_Size(int peer_) {
    int k = this->gpu_num * this->enactor->num_gpus + peer_;
    auto &enactor_slice = this->enactor->enactor_slices[k];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;
    auto request_length = frontier.output_length[0] + 2;
    auto iteration = enactor_stats.iteration;
    auto &retval = enactor_stats.retval;
    auto &graph = this->enactor->problem->sub_graphs[this->gpu_num];
    bool over_sized = false;

    if (this->enactor->flag & Debug) {
      util::PrintMsg(
          "queue_size = " + std::to_string(frontier.Next_V_Q()->GetSize()) +
              "output_length = " + std::to_string(request_length),
          this->gpu_num, iteration, peer_);
    }

    if ((this->enactor->flag & Size_Check) == 0 &&
        (this->flag & Skip_PreScan) != 0) {
      frontier.output_length[0] = 0;
    } else  // if
            // (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
    //(AdvanceKernelPolicy::ADVANCE_MODE != gunrock::oprtr::advance::LB_CULL)
    {
      retval = CheckSize<SizeT, VertexT>(
          true, "queue3", request_length, frontier.Next_V_Q(), over_sized,
          this->gpu_num, iteration, peer_, false);
      if (retval) return retval;
      retval = CheckSize<SizeT, VertexT>(true, "queue3", graph.nodes + 2,
                                         frontier.V_Q(), over_sized,
                                         this->gpu_num, iteration, peer_, true);
      if (retval) return retval;
      // TODO
      // if (enactor->problem->use_double_buffer)
      //{
      //    if (enactor_stats->retval = Check_Size<SizeT, Value>(
      //        true, "queue3", request_length,
      //        &frontier_queue->values[selector ^ 1], over_sized, thread_num,
      //        iteration, peer_, false))
      //        return;
      //    if (enactor_stats->retval = Check_Size<SizeT, Value>(
      //        true, "queue3", graph_slice->nodes + 2,
      //        &frontier_queue->values[selector], over_sized, thread_num,
      //        iteration, peer_, true))
      //        return;
      //}
    }  // else {
    //}

    return retval;
  }

  /**
   * @brief Core computation of bfs, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    static const int LOG_THREADS = 9;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &labels = data_slice.labels;
    auto &preds = data_slice.preds;
    auto &original_vertex = graph.GpT::original_vertex;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &stream = enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;
    bool debug = ((this->enactor->flag & Debug) != 0);
    bool mark_preds = ((this->enactor->problem->flag & Mark_Predecessors) != 0);
    bool idempotence =
        ((this->enactor->problem->flag & Enable_Idempotence) != 0);
    auto target = util::DEVICE;
    auto &gpu_num = this->gpu_num;

#if TO_TRACK
    util::PrintMsg(
        "Core queue_length = " + std::to_string(frontier.queue_length), gpu_num,
        iteration, peer_);
#endif
#ifdef RECORD_PER_ITERATION_STATS
    GpuTimer gpu_timer;
#endif

    if (data_slice.current_direction == FORWARD) {
      frontier.queue_reset = true;
      enactor_stats.nodes_queued[0] += frontier.queue_length;

      if (debug)
        util::PrintMsg("Forward Advance begin", gpu_num, iteration, peer_);

      LabelT label = iteration + 1;
      auto advance_op =
          [idempotence, labels, label, mark_preds, preds,
           original_vertex] __host__
          __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                     const VertexT &input_item, const SizeT &input_pos,
                     SizeT &output_pos) -> bool {
        if (!idempotence) {
          // Check if the destination node has been claimed as someone's child
          LabelT old_label = _atomicMin(labels + dest, label);
          if (label >= old_label) return false;

          // set predecessors
          if (mark_preds) {
            VertexT pred = src;
            if (original_vertex + 0 != NULL) pred = original_vertex[src];
            Store(preds + dest, pred);
          }
        }
        return true;
      };

      auto filter_op =
          [idempotence, mark_preds, preds, original_vertex] __host__ __device__(
              const VertexT &src, VertexT &dest, const SizeT &edge_id,
              const VertexT &input_item, const SizeT &input_pos,
              SizeT &output_pos) -> bool {
        if (!util::isValid(dest)) return false;

        if (idempotence && mark_preds) {
          VertexT pred = src;
          if (original_vertex + 0 != NULL) pred = original_vertex[src];
          Store(preds + dest, pred);
        }
        return true;
      };

// Edge Map
#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Start();
#endif

      auto &work_progress = frontier.work_progress;
      auto queue_index = frontier.queue_index;
      GUARD_CU(oprtr::For(
          [work_progress, queue_index] __host__ __device__(SizeT i) {
            SizeT *counter = work_progress.GetQueueCounter(queue_index + 1);
            counter[0] = 0;
          },
          1, util::DEVICE, oprtr_parameters.stream, 1, 1));
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          advance_op, filter_op));

#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Stop();
      float elapsed = gpu_timer.ElapsedMillis();
      float mteps = frontier.output_length[0] / (elapsed * 1000);
      enactor_stats.per_iteration_advance_time.push_back(elapsed);
      enactor_stats.per_iteration_advance_mteps.push_back(mteps);
      enactor_stats.per_iteration_advance_input_edges.push_back(
          frontier.queue_length);
      enactor_stats.per_iteration_advance_output_edges.push_back(
          frontier.output_length[0]);
      enactor_stats.per_iteration_advance_direction.push_back(true);
#endif
      if (debug)
        util::PrintMsg("Forward Advance end", gpu_num, iteration, peer_);

      frontier.queue_reset = false;
      // if (gunrock::oprtr::advance::hasPreScan<
      //      AdvanceKernelPolicy::ADVANCE_MODE>())
      //{
      //    enactor_stats.edges_queued[0] +=
      //        frontier_attribute.output_length[0];
      //} else {
      //    enactor_stats.AccumulateEdges(
      //    work_progress->template GetQueueLengthPointer<unsigned int>(
      //        frontier.queue_index), oprtr_parameters.tream);
      //}

      if (oprtr_parameters.advance_mode != "LB_CULL" &&
          oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
        // Filter
        if (debug) util::PrintMsg("Filter begin.", gpu_num, iteration, peer_);

        GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
            filter_op));
        if (debug) util::PrintMsg("Filter end.", gpu_num, iteration, peer_);
      }

      // Get back the resulted frontier length
      GUARD_CU(frontier.work_progress.GetQueueLength(
          frontier.queue_index, frontier.queue_length, false,
          oprtr_parameters.stream, true));
    }  // end of forward

    else {  // backward
      SizeT num_blocks = 0;
      if (data_slice.previous_direction == FORWARD) {
        GUARD_CU(data_slice.split_lengths.ForEach(
            [] __host__ __device__(SizeT & length) { length = 0; }, 1, target,
            stream));

        if (this->enactor->num_gpus == 1) {
          if (idempotence) {
            num_blocks =
                (graph.nodes >> (2 + sizeof(MaskT))) / (1 << LOG_THREADS) + 1;
            if (num_blocks > 480) num_blocks = 480;
            From_Unvisited_Queue_IDEM<ProblemT, LOG_THREADS>
                <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
                    graph.nodes,
                    data_slice.split_lengths.GetPointer(util::DEVICE),
                    data_slice.unvisited_vertices[frontier.queue_index % 2]
                        .GetPointer(util::DEVICE),
                    data_slice.visited_masks.GetPointer(util::DEVICE),
                    data_slice.labels.GetPointer(util::DEVICE));
          }

          else {
            num_blocks = graph.nodes / (1 << LOG_THREADS) + 1;
            if (num_blocks > 480) num_blocks = 480;
            From_Unvisited_Queue<ProblemT, LOG_THREADS>
                <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
                    graph.nodes,
                    data_slice.split_lengths.GetPointer(util::DEVICE),
                    data_slice.unvisited_vertices[frontier.queue_index % 2]
                        .GetPointer(util::DEVICE),
                    data_slice.labels.GetPointer(util::DEVICE));
          }
        }  // end of num_gpus == 1

        else {  // num_gpus != 1
          num_blocks =
              data_slice.local_vertices.GetSize() / (1 << LOG_THREADS) + 1;
          if (num_blocks > 480) num_blocks = 480;
          if (idempotence) {
            From_Unvisited_Queue_Local_IDEM<ProblemT, LOG_THREADS>
                <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
                    data_slice.local_vertices.GetSize(),
                    data_slice.local_vertices.GetPointer(util::DEVICE),
                    data_slice.split_lengths.GetPointer(util::DEVICE),
                    data_slice.unvisited_vertices[frontier.queue_index % 2]
                        .GetPointer(util::DEVICE),
                    data_slice.visited_masks.GetPointer(util::DEVICE),
                    data_slice.labels.GetPointer(util::DEVICE));
          }

          else {
            From_Unvisited_Queue_Local<ProblemT, LOG_THREADS>
                <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
                    data_slice.local_vertices.GetSize(),
                    data_slice.local_vertices.GetPointer(util::DEVICE),
                    data_slice.split_lengths.GetPointer(util::DEVICE),
                    data_slice.unvisited_vertices[frontier.queue_index % 2]
                        .GetPointer(util::DEVICE),
                    data_slice.labels.GetPointer(util::DEVICE));
          }
        }

        GUARD_CU(data_slice.split_lengths.Move(util::DEVICE, util::HOST, 1, 0,
                                               stream));
        GUARD_CU2(cudaStreamSynchronize(stream),
                  "cudaStreamSynchronize failed");
        data_slice.num_unvisited_vertices = data_slice.split_lengths[0];
        data_slice.num_visited_vertices =
            graph.nodes - data_slice.num_unvisited_vertices;
        if (debug)
          util::PrintMsg(
              "Changing from forward to backward, #unvisited_vertices =" +
              std::to_string(data_slice.num_unvisited_vertices));

      } else {
        data_slice.num_unvisited_vertices = data_slice.split_lengths[0];
      }

      GUARD_CU(data_slice.split_lengths.ForEach(
          [] __host__ __device__(SizeT & length) { length = 0; }, 2, target,
          stream));
      enactor_stats.nodes_queued[0] += data_slice.num_unvisited_vertices;
      num_blocks = data_slice.num_unvisited_vertices / (1 << LOG_THREADS) + 1;
      if (num_blocks > 480) num_blocks = 480;

#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Start();
#endif

      Inverse_Expand<ProblemT, LOG_THREADS>
          <<<num_blocks, 1 << LOG_THREADS, 0, stream>>>(
              graph, data_slice.num_unvisited_vertices,
              enactor_stats.iteration + 1, idempotence,
              data_slice.unvisited_vertices[frontier.queue_index % 2]
                  .GetPointer(util::DEVICE),
              data_slice.split_lengths.GetPointer(util::DEVICE),
              data_slice.unvisited_vertices[(frontier.queue_index + 1) % 2]
                  .GetPointer(util::DEVICE),
              frontier.Next_V_Q()->GetPointer(util::DEVICE),
              data_slice.visited_masks.GetPointer(util::DEVICE),
              data_slice.labels.GetPointer(util::DEVICE),
              data_slice.preds.GetPointer(util::DEVICE));

#ifdef RECORD_PER_ITERATION_STATS
      gpu_timer.Stop();
      float elapsed = gpu_timer.ElapsedMillis();
      enactor_stats.per_iteration_advance_time.push_back(elapsed);
      enactor_stats.per_iteration_advance_mteps.push_back(-1.0f);
      enactor_stats.per_iteration_advance_input_edges.push_back(-1.0f);
      enactor_stats.per_iteration_advance_output_edges.push_back(-1.0f);
      enactor_stats.per_iteration_advance_direction.push_back(false);
#endif

      data_slice.split_lengths.Move(target, util::HOST, 2, 0, stream);
      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
      if (debug)
        util::PrintMsg("#unvisited v = " +
                       std::to_string(data_slice.num_unvisited_vertices) +
                       ", #newly visited = " +
                       std::to_string(data_slice.split_lengths[1]) +
                       ", #still unvisited = " +
                       std::to_string(data_slice.split_lengths[0]));

      frontier.queue_length = data_slice.split_lengths[1];
      data_slice.num_visited_vertices =
          graph.nodes - data_slice.num_unvisited_vertices;
      enactor_stats.edges_queued[0] += frontier.output_length[0];
      frontier.queue_reset = false;
      frontier.queue_index++;
    }  // end of backward

    data_slice.previous_direction = data_slice.current_direction;

    return retval;
  }

  cudaError_t UpdatePreds(SizeT num_elements) { return cudaSuccess; }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto iteration = enactor_slice.enactor_stats.iteration;
    auto &labels = data_slice.labels;
    auto &masks = data_slice.visited_masks;
    auto &preds = data_slice.preds;
    auto label = this->enactor->mgpu_slices[this->gpu_num]
                     .in_iteration[iteration % 2][peer_];
    bool idempotence =
        ((this->enactor->problem->flag & Enable_Idempotence) != 0);
    bool mark_preds = ((ProblemT::FLAG & Mark_Predecessors) != 0);

    auto expand_op =
        [labels, label, preds, masks, idempotence, mark_preds] __host__
        __device__(VertexT & key, const SizeT &in_pos,
                   VertexT *vertex_associate_ins,
                   ValueT *value__associate_ins) -> bool {
      SizeT mask_pos = util::PreDefinedValues<SizeT>::InvalidValue;
      MaskT mask_bit = util::PreDefinedValues<MaskT>::InvalidValue;

      if (idempotence) {
        mask_pos = (key  //& (~(1ULL<<(sizeof(VertexT)*8-2)))
                         // LB_CULL::KernelPolicy::ELEMENT_ID_MASK
                    ) >>
                   (2 + sizeof(MaskT));
        MaskT mask_byte = _ldg(masks + mask_pos);
        mask_bit = 1 << (key & ((1 << (2 + sizeof(MaskT))) - 1));
        if ((mask_bit & mask_byte) != 0) return false;
      }

      if (_ldg(labels + key) != util::PreDefinedValues<LabelT>::MaxValue)
        return false;

      labels[key] = label;
      if (idempotence) masks[mask_pos] |= mask_bit;

      if (mark_preds) preds[key] = vertex_associate_ins[in_pos];
      return true;
    };

    retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of BFSIteration

/**
 * @brief BFS enactor class.
 * @tparam _Problem             BFS problem type
 * @tparam ARRAY_FLAG           Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::LabelT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  // Definations
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef typename Problem::LabelT LabelT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef BFSIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  bool direction_optimized;
  float do_a, do_b;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief BFSEnactor constructor
   */
  Enactor() : BaseEnactor("bfs"), problem(NULL) {
    this->max_num_vertex_associates =
        (Problem::FLAG & Mark_Predecessors) != 0 ? 1 : 0;
    this->max_num_value__associates = 0;
  }

  /**
   * @brief BFSEnactor destructor
   */
  virtual ~Enactor() {
    // Release();
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the enactor.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;
    util::Parameters &parameters = problem.parameters;

    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));
    direction_optimized = parameters.Get<bool>("direction-optimized");
    do_a = parameters.Get<float>("do-a");
    do_b = parameters.Get<float>("do-b");

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));

      for (int peer = 0; peer < this->num_gpus; peer++) {
        this->enactor_slices[gpu * this->num_gpus + peer]
            .oprtr_parameters.labels = &(problem.data_slices[gpu]->labels);
      }
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                [src] __host__ __device__(VertexT & v) { v = src; }, 1, target,
                0));
          }
        }
      }

      else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }
    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief one run of bfs, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        ((Enactor::Problem::FLAG & Mark_Predecessors) != 0) ? 1 : 0, 0,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a BFS computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU BFS Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};  // end of enactor

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
