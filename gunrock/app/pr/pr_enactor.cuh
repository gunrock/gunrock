// ---------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ---------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ---------------------------------------------------------------------------

/**
 * @file
 * pr_enactor.cuh
 *
 * @brief PR Problem Enactor
 */

#pragma once

#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/track_utils.cuh>
#include <gunrock/util/sort_device.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<bool>(
      "pull",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to use pull direction PageRank.", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief defination of SSSP iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct PRIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  PRIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of PageRank, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data PageRank that works on
    auto &enactor = this->enactor[0];
    auto &gpu_num = this->gpu_num;
    auto &data_slice = enactor.problem->data_slices[gpu_num][0];
    auto &enactor_slice =
        enactor.enactor_slices[gpu_num * enactor.num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &rank_curr = data_slice.rank_curr;
    auto &rank_next = data_slice.rank_next;
    auto &rank_temp = data_slice.rank_temp;
    auto &rank_temp2 = data_slice.rank_temp2;
    auto &degrees = data_slice.degrees;
    auto &local_vertices = data_slice.local_vertices;
    auto &delta = data_slice.delta;
    auto &threshold = data_slice.threshold;
    auto &reset_value = data_slice.reset_value;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;
    auto null_ptr = &local_vertices;
    null_ptr = NULL;

    if (iteration != 0) {
      if (enactor.flag & Debug)
        util::cpu_mt::PrintMessage("Filter start.", gpu_num, iteration, peer_);
      auto filter_op =
          [rank_curr, rank_next, degrees, delta, threshold,
           reset_value] __host__
          __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                     const VertexT &input_item, const SizeT &input_pos,
                     SizeT &output_pos) -> bool {
        ValueT old_value = rank_curr[dest];
        ValueT new_value = delta * rank_next[dest];
        new_value = reset_value + new_value;
        if (degrees[dest] != 0) new_value /= degrees[dest];
        if (!isfinite(new_value)) new_value = 0;
        rank_curr[dest] = new_value;
        // if (util::isTracking(dest))
        //    printf("rank[%d] = %f -> %f = (%f + %f * %f) / %d\n",
        //        dest, old_value, new_value, reset_value,
        //        delta, rank_next[dest], degrees[dest]);
        return (fabs(new_value - old_value) > (threshold * old_value));
      };

      frontier.queue_length = data_slice.local_vertices.GetSize();
      enactor_stats.nodes_queued[0] += frontier.queue_length;
      frontier.queue_reset = true;
      oprtr_parameters.filter_mode = "BY_PASS";

      // filter kernel
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.coo(), &local_vertices, null_ptr, oprtr_parameters, filter_op));

      if (enactor.flag & Debug)
        util::cpu_mt::PrintMessage("Filter end.", gpu_num, iteration, peer_);

      frontier.queue_index++;
      // Get back the resulted frontier length
      GUARD_CU(frontier.work_progress.GetQueueLength(
          frontier.queue_index, frontier.queue_length, false,
          oprtr_parameters.stream, true));

      if (!data_slice.pull) {
        GUARD_CU(rank_next.ForEach(
            [] __host__ __device__(ValueT & rank) { rank = 0.0; }, graph.nodes,
            util::DEVICE, oprtr_parameters.stream));
      }

      GUARD_CU2(cudaStreamSynchronize(oprtr_parameters.stream),
                "cudaStreamSynchronize failed");
      data_slice.num_updated_vertices = frontier.queue_length;
    }

    if (data_slice.pull) {
      if (enactor.flag & Debug)
        util::cpu_mt::PrintMessage("NeighborReduce start.", gpu_num, iteration,
                                   peer_);

      auto advance_op =
          [rank_curr, graph] __host__ __device__(
              const VertexT &src, VertexT &dest, const SizeT &edge_id,
              const VertexT &input_item, const SizeT &input_pos,
              SizeT &output_pos) -> ValueT { return rank_curr[dest]; };

      oprtr_parameters.reduce_values_out = &rank_next;
      oprtr_parameters.reduce_reset = true;
      oprtr_parameters.reduce_values_temp = &rank_temp;
      oprtr_parameters.reduce_values_temp2 = &rank_temp2;
      oprtr_parameters.advance_mode = "ALL_EDGES";
      frontier.queue_length = graph.nodes;
      frontier.queue_reset = true;
      GUARD_CU(oprtr::NeighborReduce<oprtr::OprtrType_V2V |
                                     oprtr::OprtrMode_REDUCE_TO_SRC |
                                     oprtr::ReduceOp_Plus>(
          graph.csc(), null_ptr, null_ptr, oprtr_parameters, advance_op,
          [] __host__ __device__(const ValueT &a, const ValueT &b) {
            return a + b;
          },
          (ValueT)0));
    }

    else {
      if (enactor.flag & Debug)
        util::cpu_mt::PrintMessage("Advance start.", gpu_num, iteration, peer_);

      auto advance_op = [rank_curr, rank_next] __host__ __device__(
                            const VertexT &src, VertexT &dest,
                            const SizeT &edge_id, const VertexT &input_item,
                            const SizeT &input_pos, SizeT &output_pos) -> bool {
        // printf("%d -> %d\n", src, dest);
        ValueT add_value = rank_curr[src];
        if (isfinite(add_value)) {
          atomicAdd(rank_next + dest, add_value);
          // ValueT old_val = atomicAdd(rank_next + dest, add_value);
          // if (dest == 42029)
          //    printf("rank[%d] = %f = %f (rank[%d]) + %f\n",
          //        dest, old_val + add_value, add_value, src, old_val);
        }
        return true;
      };

      // Edge Map
      frontier.queue_length = local_vertices.GetSize();
      frontier.queue_reset = true;
      oprtr_parameters.advance_mode = "ALL_EDGES";
      GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.coo(), &local_vertices, null_ptr, oprtr_parameters,
          advance_op));
    }

    enactor_stats.edges_queued[0] += graph.edges;
    return retval;
  }

  cudaError_t Compute_OutputLength(int peer_) {
    // No need to load balance or get output size
    return cudaSuccess;
  }

  cudaError_t Check_Queue_Size(int peer_) {
    // no need to check queue size for PR
    return cudaSuccess;
  }

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
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &rank_next = data_slice.rank_next;

    auto expand_op = [rank_next] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      ValueT in_val = value__associate_ins[in_pos];
      atomicAdd(rank_next + key, in_val);
      return false;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  cudaError_t UpdatePreds(SizeT num_elements) {
    // No need to update predecessors
    return cudaSuccess;
  }

  /*
   * @brief Make_Output function.
   * @tparam NUM_VERTEX_ASSOCIATES
   * @tparam NUM_VALUE__ASSOCIATES
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t MakeOutput(SizeT num_elements) {
    cudaError_t retval = cudaSuccess;
    int num_gpus = this->enactor->num_gpus;
    int gpu_num = this->gpu_num;
    auto &enactor = this->enactor[0];
    auto &enactor_slice =
        enactor.enactor_slices[gpu_num * num_gpus +
                               ((enactor.flag & Size_Check) ? 0 : num_gpus)];
    auto &mgpu_slice = enactor.mgpu_slices[gpu_num];
    auto &data_slice = enactor.problem->data_slices[gpu_num][0];
    auto &rank_next = data_slice.rank_next;
    cudaStream_t stream = enactor_slice.stream;

    if (num_gpus < 2) return retval;

    for (int peer_ = 1; peer_ < num_gpus; peer_++) {
      auto &remote_vertices_out = data_slice.remote_vertices_out[peer_];
      mgpu_slice.out_length[peer_] = remote_vertices_out.GetSize();
      GUARD_CU(mgpu_slice.value__associate_out[peer_].ForAll(
          [remote_vertices_out, rank_next] __host__ __device__(
              ValueT * values_out, const SizeT &pos) {
            values_out[pos] = rank_next[remote_vertices_out[pos]];
          },
          mgpu_slice.out_length[peer_], util::DEVICE, stream));
    }
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &enactor_slices = this->enactor->enactor_slices;
    int num_gpus = this->enactor->num_gpus;
    for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
      auto &retval = enactor_slices[gpu].enactor_stats.retval;
      if (retval == cudaSuccess) continue;
      printf("(CUDA error %d @ GPU %d: %s\n", retval, gpu % num_gpus,
             cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    auto &data_slices = this->enactor->problem->data_slices;
    bool all_zero = true;
    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (data_slices[gpu]->num_updated_vertices)  // PR_queue_length > 0)
      {
        // printf("data_slice[%d].PR_queue_length = %d\n", gpu,
        // data_slice[gpu]->PR_queue_length);
        all_zero = false;
      }
    if (all_zero) return true;

    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (enactor_slices[gpu * num_gpus].enactor_stats.iteration <
          data_slices[0]->max_iter) {
        // printf("enactor_stats[%d].iteration = %lld\n", gpu, enactor_stats[gpu
        // * num_gpus].iteration);
        return false;
      }
    return true;
  }
};

/**
 * @brief PageRank enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor : public EnactorBase<typename _Problem::GraphT,
                                   typename _Problem::VertexT,  // LabelT
                                   typename _Problem::ValueT, ARRAY_FLAG,
                                   cudaHostRegisterFlag> {
 public:
  // Definations
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef PRIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  // Methods
  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief PREnactor constructor
   */
  Enactor() : BaseEnactor("pr"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   *  @brief PREnactor destructor
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

    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    if (this->num_gpus == 1) {
      GUARD_CU(this->Init_Threads(
          this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
      return retval;
    }

    auto &data_slices = problem.data_slices;
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      auto &data_slice_l = data_slices[gpu][0];
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      for (int peer = 0; peer < this->num_gpus; peer++) {
        if (peer == gpu) continue;
        int peer_ = (peer < gpu) ? peer + 1 : peer;
        int gpu_ = (peer < gpu) ? gpu : gpu + 1;
        auto &data_slice_p = data_slices[peer][0];

        data_slice_l.in_counters[peer_] = data_slice_p.out_counters[gpu_];
        if (gpu != 0) {
          data_slice_l.remote_vertices_in[peer_].SetPointer(
              data_slice_p.remote_vertices_out[gpu_].GetPointer(util::HOST),
              data_slice_p.remote_vertices_out[gpu_].GetSize(), util::HOST);
        } else {
          data_slice_l.remote_vertices_in[peer_].SetPointer(
              data_slice_p.remote_vertices_out[gpu_].GetPointer(util::HOST),
              max(data_slice_p.remote_vertices_out[gpu_].GetSize(),
                  data_slice_p.local_vertices.GetSize()),
              util::HOST);
        }
        GUARD_CU(data_slice_l.remote_vertices_in[peer_].Move(
            util::HOST, target,
            data_slice_p.remote_vertices_out[gpu_].GetSize()));

        for (int t = 0; t < 2; t++) {
          GUARD_CU(
              this->mgpu_slices[gpu].value__associate_in[t][peer_].EnsureSize_(
                  data_slice_l.in_counters[peer_], target));
        }
      }
    }

    for (int gpu = 1; gpu < this->num_gpus; gpu++) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(this->mgpu_slices[gpu].value__associate_out[1].EnsureSize_(
          problem.data_slices[gpu]->local_vertices.GetSize(), target));
    }

    if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[0]));
    for (int peer = 1; peer < this->num_gpus; peer++) {
      GUARD_CU(this->mgpu_slices[0].value__associate_in[0][peer].EnsureSize_(
          problem.data_slices[peer]->local_vertices.GetSize(), target));
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
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      /*thread_slices[gpu].status = ThreadSlice::Status::Wait;

      if (retval = util::SetDevice(problem -> gpu_idx[gpu]))
          return retval;
      if (AdvanceKernelPolicy::ADVANCE_MODE ==
      gunrock::oprtr::advance::TWC_FORWARD)
      {
          //return retval;
      } else {
          bool over_sized = false;
          if (retval = Check_Size<SizeT, SizeT> (
              this -> size_check, "scanned_edges",
              problem -> data_slices[gpu] -> local_vertices.GetSize() + 2,
              problem -> data_slices[gpu] -> scanned_edges,
              over_sized, -1, -1, -1, false)) return retval;
          this -> frontier_attribute [gpu * this -> num_gpus].queue_length
              = problem -> data_slices[gpu] -> local_vertices.GetSize();

          retval = gunrock::oprtr::advance::ComputeOutputLength
              <AdvanceKernelPolicy, Problem, PRFunctor<VertexId, SizeT, Value,
      Problem>, gunrock::oprtr::advance::V2V>( this -> frontier_attribute + gpu
      * this -> num_gpus,//frontier_attribute, problem -> graph_slices[gpu] ->
      row_offsets.GetPointer(util::DEVICE),//d_offsets, problem ->
      graph_slices[gpu] -> column_indices.GetPointer(util::DEVICE),//d_indices,
              (SizeT   *)NULL, ///d_inv_offsets,
              (VertexId*)NULL,//d_inv_indices,
              problem -> data_slices[gpu] ->
      local_vertices.GetPointer(util::DEVICE),//d_in_key_queue, problem ->
      data_slices[gpu] ->
      scanned_edges[0].GetPointer(util::DEVICE),//partitioned_scanned_edges->GetPointer(util::DEVICE),
              problem -> graph_slices[gpu] -> nodes,//max_in,
              problem -> graph_slices[gpu] -> edges,//max_out,
              thread_slices[gpu].context[0][0],
              problem -> data_slices[gpu] -> streams[0],
              //ADVANCE_TYPE,
              false,
              false,
              false);

          if (retval = this -> frontier_attribute[gpu * this ->
      num_gpus].output_length.Move(util::DEVICE, util::HOST, 1, 0, problem ->
      data_slices[gpu] -> streams[0])) return retval; if (retval =
      util::GRError(cudaStreamSynchronize(problem -> data_slices[gpu] ->
      streams[0]), "cudaStreamSynchronize failed", __FILE__, __LINE__)) return
      retval;
      }*/

      for (int peer = 0; peer < this->num_gpus; peer++) {
        auto &frontier =
            this->enactor_slices[gpu * this->num_gpus + peer].frontier;

        frontier.queue_length =
            (peer != 0)
                ? 0
                : this->problem->data_slices[gpu]->local_vertices.GetSize();
        frontier.queue_index = 0;  // Work queue index
        frontier.queue_reset = true;
        this->enactor_slices[gpu * this->num_gpus + peer]
            .enactor_stats.iteration = 0;
      }

      if (this->num_gpus > 1) {
        if (target & util::DEVICE)
          GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

        this->mgpu_slices[gpu].value__associate_orgs[0] =
            this->problem->data_slices[gpu]->rank_next.GetPointer(target);
        GUARD_CU(this->mgpu_slices[gpu].value__associate_orgs.Move(util::HOST,
                                                                   target));
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief one run of sssp, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 1, IterationT>(
        thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a PR computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU PageRank Done.", this->flag & Debug);
    return retval;
  }

  cudaError_t Extract() {
    cudaError_t retval = cudaSuccess;
    auto &data_slices = this->problem->data_slices;
    int num_gpus = this->num_gpus;
    for (int gpu_num = 1; gpu_num < num_gpus; gpu_num++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu_num]));
      auto &data_slice = data_slices[gpu_num][0];
      auto &enactor_slice = this->enactor_slices[gpu_num * num_gpus];
      auto &degrees = data_slice.degrees;
      auto &rank_curr = data_slice.rank_curr;
      auto &rank_out = this->mgpu_slices[gpu_num].value__associate_out[1];
      auto &enactor_stats = enactor_slice.enactor_stats;
      auto &stream = enactor_slice.stream2;

      GUARD_CU(data_slice.local_vertices.ForAll(
          [rank_curr, degrees, rank_out] __host__ __device__(VertexT * vertices,
                                                             const SizeT &pos) {
            VertexT v = vertices[pos];
            ValueT rank = rank_curr[v];
            if (degrees[v] != 0) rank *= degrees[v];
            rank_out[pos] = rank;
          },
          data_slice.local_vertices.GetSize(), util::DEVICE, stream));

      enactor_stats.iteration = 0;
      PushNeighbor<EnactorT, 0, 1>(*this, gpu_num, 0);
      SetRecord(this->mgpu_slices[gpu_num], enactor_stats.iteration, 1, 0,
                stream);
      data_slice.final_event_set = true;
    }

    GUARD_CU(util::SetDevice(this->gpu_idx[0]));
    auto &data_slice = data_slices[0][0];
    auto &enactor_slice = this->enactor_slices[0];
    auto &degrees = data_slice.degrees;
    auto &rank_curr = data_slice.rank_curr;
    auto &stream = enactor_slice.stream2;
    GUARD_CU(data_slice.local_vertices.ForAll(
        [rank_curr, degrees] __host__ __device__(VertexT * vertices,
                                                 const SizeT &pos) {
          VertexT v = vertices[pos];
          ValueT rank = rank_curr[v];
          if (degrees[v] != 0) rank *= degrees[v];
          rank_curr[v] = rank;
        },
        data_slice.local_vertices.GetSize(), util::DEVICE, stream));

    for (int peer = 1; peer < num_gpus; peer++) {
      GUARD_CU2(
          cudaMemcpyAsync(
              data_slice.remote_vertices_in[peer].GetPointer(util::DEVICE),
              data_slices[peer]->local_vertices.GetPointer(util::HOST),
              sizeof(VertexT) * data_slices[peer]->local_vertices.GetSize(),
              cudaMemcpyHostToDevice, this->enactor_slices[peer].stream),
          "cudaMemcpyAsync failed");
    }

    for (int peer = 1; peer < num_gpus; peer++) {
      int peer_iteration =
          this->enactor_slices[peer * num_gpus].enactor_stats.iteration;
      GUARD_CU2(
          cudaStreamWaitEvent(
              this->enactor_slices[peer].stream,
              this->mgpu_slices[peer].events[peer_iteration % 4][0][0], 0),
          "cudaStreamWaitEvent failed");

      auto &rank_in =
          this->mgpu_slices[0].value__associate_in[peer_iteration % 2][peer];
      GUARD_CU(data_slice.remote_vertices_in[peer].ForAll(
          [rank_curr, rank_in] __host__ __device__(VertexT * keys_in,
                                                   SizeT & pos) {
            VertexT v = keys_in[pos];
            rank_curr[v] = rank_in[pos];
          },
          data_slices[peer]->local_vertices.GetSize(), util::DEVICE,
          this->enactor_slices[peer].stream));

      GUARD_CU2(
          cudaEventRecord(
              this->mgpu_slices[0]
                  .events[enactor_slice.enactor_stats.iteration % 4][peer][0],
              this->enactor_slices[peer].stream),
          "cudaEventRecord failed");
      GUARD_CU2(
          cudaStreamWaitEvent(
              this->enactor_slices[0].stream,
              this->mgpu_slices[0]
                  .events[enactor_slice.enactor_stats.iteration % 4][peer][0],
              0),
          "cudaStreamWaitEvent failed");
    }

    SizeT nodes = data_slice.org_nodes;
    GUARD_CU(data_slice.node_ids.EnsureSize_(nodes, util::DEVICE));
    GUARD_CU(data_slice.temp_vertex.EnsureSize_(nodes, util::DEVICE));

    GUARD_CU(data_slice.node_ids.ForAll(
        [] __host__ __device__(VertexT * ids, const SizeT &pos) {
          ids[pos] = pos;
        },
        nodes, util::DEVICE, this->enactor_slices[0].stream));

    // util::PrintMsg("#nodes = " + std::to_string(nodes));
    /*size_t cub_required_size = 0;
    void* temp_storage = NULL;
    cub::DoubleBuffer<ValueT > key_buffer(
        data_slice.rank_curr.GetPointer(util::DEVICE),
        data_slice.rank_next.GetPointer(util::DEVICE));
    cub::DoubleBuffer<VertexT> value_buffer(
        data_slice.node_ids   .GetPointer(util::DEVICE),
        data_slice.temp_vertex.GetPointer(util::DEVICE));
    GUARD_CU2(cub::DeviceRadixSort::SortPairsDescending(
        temp_storage, cub_required_size,
        key_buffer, value_buffer, nodes,
        0, sizeof(ValueT) * 8, this -> enactor_slices[0].stream),
        "cubDeviceRadixSort failed");
    GUARD_CU(data_slice.cub_sort_storage.EnsureSize_(
        cub_required_size, util::DEVICE));

    GUARD_CU2(cudaDeviceSynchronize(),
        "cudaDeviceSynchronize failed.");

    printf("cub_sort_stoarge = %p, size = %d\n",
        data_slice.cub_sort_storage.GetPointer(util::DEVICE),
        data_slice.cub_sort_storage.GetSize());

    // sort according to the rank of nodes
    GUARD_CU2(cub::DeviceRadixSort::SortPairsDescending(
        data_slice.cub_sort_storage.GetPointer(util::DEVICE),
        cub_required_size,
        key_buffer, value_buffer, nodes,
        0, sizeof(ValueT) * 8, this -> enactor_slices[0].stream),
        "cubDeviceRadixSort failed");

    GUARD_CU2(cudaDeviceSynchronize(),
        "cudaDeviceSynchronize failed.");

    if (key_buffer.Current() != data_slice.rank_curr.GetPointer(util::DEVICE))
    {
        ValueT *keys = key_buffer.Current();
        GUARD_CU(data_slice.rank_curr.ForEach(keys,
            []__host__ __device__(ValueT &rank, const ValueT &key)
            {
                rank = key;
            }, nodes, util::DEVICE, this -> enactor_slices[0].stream));
    }

    if (value_buffer.Current() != data_slice.node_ids.GetPointer(util::DEVICE))
    {
        VertexT *values = value_buffer.Current();
        GUARD_CU(data_slice.node_ids.ForEach(values,
            []__host__ __device__(VertexT &node_id, const VertexT &val)
            {
                node_id = val;
            }, nodes, util::DEVICE, this -> enactor_slices[0].stream));
    }*/

    // util::Array1D<SizeT, char> cub_temp_space;
    GUARD_CU(util::cubSortPairsDescending(
        data_slice.cub_sort_storage, data_slice.rank_curr, data_slice.rank_next,
        data_slice.node_ids, data_slice.temp_vertex, nodes, 0,
        sizeof(ValueT) * 8, this->enactor_slices[0].stream));

    // GUARD_CU2(cudaDeviceSynchronize(),
    //    "cudaDeviceSynchronize failed.");

    auto &temp_vertex = data_slice.temp_vertex;
    // auto &rank_curr   = data_slice.rank_curr;
    auto &rank_next = data_slice.rank_next;
    GUARD_CU(data_slice.node_ids.ForAll(
        [temp_vertex, rank_curr, rank_next] __host__ __device__(
            VertexT * ids, const SizeT &v) {
          ids[v] = temp_vertex[v];
          rank_curr[v] = rank_next[v];
        },
        nodes, util::DEVICE, this->enactor_slices[0].stream));

    if (data_slice.scale) {
      ValueT a = 1.0 / (ValueT)nodes;
      GUARD_CU(data_slice.rank_curr.ForEach(
          [a] __host__ __device__(ValueT & rank) { rank *= a; }, nodes,
          util::DEVICE, this->enactor_slices[0].stream));
    }

    GUARD_CU2(cudaStreamSynchronize(this->enactor_slices[0].stream),
              "cudaStreamSynchronize failed");
    return retval;
  }

  /** @} */
};

}  // namespace pr
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
