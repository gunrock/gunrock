// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_enactor.cuh
 *
 * @brief BC Problem Enactor
 */

#pragma once

#include <iostream>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Speciflying parameters for BC Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of BC iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct BCForwardIterationLoop
    : public IterationLoopBase<EnactorT,
                               Use_FullQ | Push>  //| Update_Predecessors>
{
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push  //| Update_Predecessors
                            >
      BaseIterationLoop;

  BCForwardIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of bc, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data alias the enactor works on
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;

    // BC specific data alias here, e.g.:
    auto &sigmas = data_slice.sigmas;
    auto &labels = data_slice.labels;

    // ----------------------------
    // Forward advance -- BFS

    auto advance_op = [labels, sigmas] __host__ __device__(
                          const VertexT &src, VertexT &dest,
                          const SizeT &edge_id, const VertexT &input_item,
                          const SizeT &input_pos, SizeT &output_pos) -> bool {
      // Check if the destination node has been claimed as someone's child
      VertexT new_label = Load<cub::LOAD_CG>(labels + src) + 1;
      VertexT old_label =
          atomicCAS(labels + dest,
                    util::PreDefinedValues<VertexT>::InvalidValue, new_label);
      if (old_label != new_label && util::isValid(old_label)) return false;

      // Accumulate sigma value
      atomicAdd(sigmas + dest, sigmas[src]);
      if (!util::isValid(old_label)) {
        return true;
      } else {
        return false;
      }
    };

    auto filter_op = [] __host__ __device__(
                         const VertexT &src, VertexT &dest,
                         const SizeT &edge_id, const VertexT &input_item,
                         const SizeT &input_pos, SizeT &output_pos) -> bool {
      return util::isValid(dest);
    };

    // Call the advance operator, using the advance operation.
    // BC only uses an advance + a filter, with
    // possible optimization to fuze the two kernels.
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        advance_op, filter_op));

    if (oprtr_parameters.advance_mode != "LB_CULL" &&
        oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
      frontier.queue_reset = false;
      // Call the filter operator, using the filter operation
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          filter_op));
    }

    // Get back the resulted frontier length
    GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false,
        oprtr_parameters.stream, true));

    return retval;
  }

  cudaError_t Gather(int peer_) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;

    if (enactor_stats.iteration <= 0) return retval;

    SizeT cur_offset = data_slice.forward_queue_offsets[peer_].back();
    bool over_sized = false;
    retval = CheckSize<SizeT, VertexT>(
        (this->enactor->flag & Size_Check) != 0, "forward_output",
        cur_offset + frontier.queue_length, &data_slice.forward_output[peer_],
        over_sized, this->gpu_num, enactor_stats.iteration, peer_);
    if (retval) return retval;

    auto &forward_output = data_slice.forward_output[peer_];
    GUARD_CU(frontier.V_Q()->ForAll(
        [forward_output, cur_offset] __host__ __device__(const VertexT *v_q,
                                                         const SizeT &i) {
          forward_output[cur_offset + i] = v_q[i];
        },
        frontier.queue_length, util::DEVICE, oprtr_parameters.stream));

    data_slice.forward_queue_offsets[peer_].push_back(frontier.queue_length +
                                                      cur_offset);
    return retval;
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
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distances
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of BCForwardIterationLoop

template <typename EnactorT>
struct BCBackwardIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Pull> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;

  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;

  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Pull> BaseIterationLoop;

  BCBackwardIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of bc, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data alias the enactor works on
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;

    // BC problem specific data alias here, e.g.:
    auto &bc_values = data_slice.bc_values;
    auto &sigmas = data_slice.sigmas;
    auto &deltas = data_slice.deltas;
    auto &labels = data_slice.labels;
    auto src_node = data_slice.src_node;
    auto num_vertices = graph.nodes;

    // ----------------------------
    // Backward advance -- accumulating BC values

    auto advance_op =
        [labels, deltas, bc_values, iteration, src_node, sigmas,
         num_vertices] __host__
        __device__(const VertexT &src, VertexT &dest, const SizeT &edge_id,
                   const VertexT &input_item, const SizeT &input_pos,
                   SizeT &output_pos) -> bool {
      VertexT s_label = Load<cub::LOAD_CG>(labels + src);
      VertexT d_label = Load<cub::LOAD_CG>(labels + dest);

      if (iteration == 0) {
        return (d_label == s_label + 1);
      } else {
        if (d_label == s_label + 1) {
          if (src == src_node) return true;  // !! Is this right? YC: it's right

          ValueT from_sigma = Load<cub::LOAD_CG>(sigmas + src);
          ValueT to_sigma = Load<cub::LOAD_CG>(sigmas + dest);
          ValueT to_delta = Load<cub::LOAD_CG>(deltas + dest);
          ValueT result = from_sigma / to_sigma * (1.0 + to_delta);

          // Accumulate bc value
          ValueT old_delta = atomicAdd(deltas + src, result);
          ValueT old_bc_value = atomicAdd(bc_values + src, result);
          return true;
        } else {
          return false;
        }
      }
    };

    auto filter_op = [labels] __host__ __device__(
                         const VertexT &src, VertexT &dest,
                         const SizeT &edge_id, const VertexT &input_item,
                         const SizeT &input_pos, SizeT &output_pos) -> bool {
      return labels[dest] == 0;
    };

    frontier.queue_reset = true;
    auto empty_q = frontier.Next_V_Q();
    empty_q = NULL;
    // Call the advance operator, using the advance operation.
    // BC only uses an advance + a filter, with
    // possible optimization to fuze the two kernels.

    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(graph.csr(), frontier.V_Q(),
                                                  empty_q, oprtr_parameters,
                                                  advance_op, filter_op));

    if (oprtr_parameters.advance_mode != "LB_CULL" &&
        oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
      frontier.queue_reset = false;
      // Call the filter operator, using the filter operation
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), empty_q, oprtr_parameters, filter_op));
    }

    // Get back the resulted frontier length
    GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false,
        oprtr_parameters.stream, true));

    return retval;
  }

  cudaError_t Change() {
    auto &enactor_stats =
        this->enactor->enactor_slices[this->gpu_num * this->enactor->num_gpus]
            .enactor_stats;
    enactor_stats.iteration--;
    return enactor_stats.retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &enactor_slices = this->enactor->enactor_slices;
    auto iter = enactor_slices[0].enactor_stats.iteration;
    if (All_Done(this->enactor[0], gpu_num)) {
      if (iter > 1) {
        return false;
      } else {
        return true;
      }
    } else {
      if (iter < 0) {
        return true;
      } else {
        return false;
      }
    }
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
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distances
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  cudaError_t Gather(int peer_) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;

    // printf("-- Gather --\n");
    // printf("  queue_length=%d\n", frontier.queue_length);

    SizeT cur_pos = data_slice.forward_queue_offsets[peer_].back();
    data_slice.forward_queue_offsets[peer_].pop_back();
    SizeT pre_pos = data_slice.forward_queue_offsets[peer_].back();
    frontier.queue_reset = true;
    if (enactor_stats.iteration > 0 && cur_pos - pre_pos > 0) {
      frontier.queue_length = cur_pos - pre_pos;

      bool over_sized = false;
      // if (enactor_stats -> retval = Check_Size<SizeT, VertexId> (
      //     enactor -> size_check, "queue1",
      //     frontier.queue_length,
      //     &frontier_queue -> keys[frontier_queue -> selector],
      //     over_sized, thread_num, enactor_stats->iteration, peer_, false))
      //     return;
      retval = CheckSize<SizeT, VertexT>(
          (this->enactor->flag & Size_Check) != 0, "queue1",
          frontier.queue_length, frontier.V_Q(), over_sized, this->gpu_num,
          enactor_stats.iteration, peer_);
      if (retval) return retval;

      // MemsetCopyVectorKernel<<<120, 512, 0, oprtr_parameters.stream>>>(
      //    frontier.V_Q()->GetPointer(util::DEVICE),
      //    data_slice.forward_output[peer_].GetPointer(util::DEVICE) + pre_pos,
      //    frontier.queue_length);
      auto &forward_output = data_slice.forward_output[peer_];
      GUARD_CU(frontier.V_Q()->ForAll(
          [forward_output, pre_pos] __host__ __device__(VertexT * v_q,
                                                        const SizeT &i) {
            v_q[i] = forward_output[pre_pos + i];
          },
          frontier.queue_length, util::DEVICE, oprtr_parameters.stream));

    } else {
      frontier.queue_length = 0;
    }
    return retval;
  }
};  // end of BCBackwardIterationLoop

/**
 * @brief BC enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT
      LabelT;  // e.g. typedef typename Problem::LabelT LabelT;
  typedef typename Problem::ValueT
      ValueT;  // e.g. typedef typename Problem::ValueT ValueT;
  typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;

  typedef BCForwardIterationLoop<EnactorT> ForwardIterationT;
  typedef BCBackwardIterationLoop<EnactorT> BackwardIterationT;

  Problem *problem;
  ForwardIterationT *forward_iterations;
  BackwardIterationT *backward_iterations;

  /**
   * @brief BCEnactor constructor
   */
  Enactor() : BaseEnactor("bc"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 2;
  }

  /**
   * @brief BCEnactor destructor
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
    delete[] forward_iterations;
    forward_iterations = NULL;
    delete[] backward_iterations;
    backward_iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Initialize the problem.
   * @param[in] parameters Running parameters.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target,
                               false));  // 2 vertex frontiers

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

    forward_iterations = new ForwardIterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(forward_iterations[gpu].Init(this, gpu));
    }

    backward_iterations = new BackwardIterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(backward_iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of BC, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 1, ForwardIterationT>(
        thread_data, forward_iterations[thread_data.thread_num]);
    gunrock::app::Iteration_Loop<0, 2, BackwardIterationT>(
        thread_data, backward_iterations[thread_data.thread_num]);
    return cudaSuccess;
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
   * @brief Enacts a BC computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(VertexT src) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU BC Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace bc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
