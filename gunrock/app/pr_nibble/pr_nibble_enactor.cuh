// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * Template_enactor.cuh
 *
 * @brief pr_nibble Problem Enactor
 */

#pragma once

#include <iostream>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/pr_nibble/pr_nibble_problem.cuh>

namespace gunrock {
namespace app {
namespace pr_nibble {

/**
 * @brief Speciflying parameters for pr_nibble Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of pr_nibble iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct PRNibbleIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  PRNibbleIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of pr_nibble, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // --
    // Alias variables

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

    // problem specific data alias
    auto &grad = data_slice.grad;
    auto &q = data_slice.q;
    auto &y = data_slice.y;
    auto &z = data_slice.z;
    auto &touched = data_slice.touched;

    auto &alpha = data_slice.alpha;
    auto &rho = data_slice.rho;
    auto &eps = data_slice.eps;
    auto &max_iter = data_slice.max_iter;

    auto &src_node = data_slice.src;
    auto &src_neib = data_slice.src_neib;
    auto &num_ref_nodes = data_slice.num_ref_nodes;

    auto &d_grad_scale = data_slice.d_grad_scale;
    auto &d_grad_scale_value = data_slice.d_grad_scale_value;

    // --
    // Define operations

    // compute operation
    auto compute_op = [graph, iteration, src_node, src_neib, z, y, grad, q,
                       alpha, rho, touched, num_ref_nodes] __host__
                      __device__(VertexT * v, const SizeT &i) {
                        VertexT idx = v[i];

                        // ignore the neighbor on the first iteration
                        if ((iteration == 0) && (idx == src_neib)) return;

                        // Compute degrees
                        SizeT idx_d = graph.GetNeighborListLength(idx);
                        ValueT idx_d_sqrt = sqrt((ValueT)idx_d);
                        ValueT idx_dn_sqrt = 1.0 / idx_d_sqrt;

                        // this is at end in original implementation, but works
                        // here after the first iteration (+ have to adjust for
                        // it in StopCondition)
                        if ((iteration > 0) && (idx == src_node)) {
                          grad[idx] -= alpha / num_ref_nodes * idx_dn_sqrt;
                        }

                        z[idx] = y[idx] - grad[idx];

                        if (z[idx] == 0) return;

                        ValueT q_old = q[idx];
                        ValueT thresh = rho * alpha * idx_d_sqrt;

                        if (z[idx] >= thresh) {
                          q[idx] = z[idx] - thresh;
                        } else if (z[idx] <= -thresh) {
                          q[idx] = z[idx] + thresh;
                        } else {
                          q[idx] = (ValueT)0;
                        }

                        if (iteration == 0) {
                          y[idx] = q[idx];
                        } else {
                          ValueT beta = (1 - sqrt(alpha)) / (1 + sqrt(alpha));
                          y[idx] = q[idx] + beta * (q[idx] - q_old);
                        }

                        touched[idx] = 0;
                        grad[idx] = y[idx] * (1.0 + alpha) / 2;
                      };

    GUARD_CU(frontier.V_Q()->ForAll(compute_op, frontier.queue_length,
                                    util::DEVICE, oprtr_parameters.stream));

    // advance operation
    auto advance_op =
        [graph, touched, grad, y, alpha, iteration] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      ValueT src_dn_sqrt = 1.0 / sqrt((ValueT)graph.GetNeighborListLength(src));
      ValueT dest_dn_sqrt =
          1.0 / sqrt((ValueT)graph.GetNeighborListLength(dest));
      ValueT src_y = Load<cub::LOAD_CG>(y + src);

      ValueT grad_update =
          -src_dn_sqrt * src_y * dest_dn_sqrt * (1.0 - alpha) / 2;
      ValueT last_grad = atomicAdd(grad + dest, grad_update);
      if (last_grad + grad_update == 0) return false;

      bool already_touched = atomicMax(touched + dest, 1) == 1;
      return !already_touched;
    };

    // filter operation
    auto filter_op = [] __host__ __device__(
                         const VertexT &src, VertexT &dest,
                         const SizeT &edge_id, const VertexT &input_item,
                         const SizeT &input_pos,
                         SizeT &output_pos) -> bool { return true; };

    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
        advance_op, filter_op));

    if (oprtr_parameters.advance_mode != "LB_CULL" &&
        oprtr_parameters.advance_mode != "LB_LIGHT_CULL") {
      frontier.queue_reset = false;
      GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          filter_op));
    }

    GUARD_CU(frontier.work_progress.GetQueueLength(
        frontier.queue_index, frontier.queue_length, false,
        oprtr_parameters.stream, true));

    // Convergence checking
    ValueT grad_thresh = rho * alpha * (1 + eps);
    GUARD_CU(cudaMemset(d_grad_scale, 0, 1 * sizeof(int)));
    GUARD_CU(cudaMemset(d_grad_scale_value, 0, 1 * sizeof(ValueT)));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    auto convergence_op = [graph, grad, d_grad_scale, d_grad_scale_value,
                           grad_thresh, iteration, src_node, alpha,
                           num_ref_nodes] __host__ __device__(VertexT & v) {
      ValueT v_dn_sqrt = 1.0 / sqrt((ValueT)graph.GetNeighborListLength(v));

      ValueT val = grad[v];
      if (v == src_node) val -= (alpha / num_ref_nodes) * v_dn_sqrt;

      val = abs(val * v_dn_sqrt);

      atomicMax(d_grad_scale_value, val);
      if (val > grad_thresh) {
        atomicMax(d_grad_scale, 1);
      }
    };

    GUARD_CU(frontier.V_Q()->ForEach(convergence_op, frontier.queue_length,
                                     util::DEVICE, oprtr_parameters.stream));

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    cudaMemcpy(data_slice.h_grad_scale, d_grad_scale, 1 * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(data_slice.h_grad_scale_value, d_grad_scale_value,
               1 * sizeof(ValueT), cudaMemcpyDeviceToHost);
    // printf("data_slice.h_grad_scale=%d | h_val=%0.17g |
    // grad_thresh=%0.17g\n",
    //     data_slice.h_grad_scale[0], data_slice.h_grad_scale_value[0],
    //     grad_thresh);

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &enactor_slice = this->enactor->enactor_slices[0];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &iter = enactor_stats.iteration;

    // never break on first iteration
    if (iter == 0) {
      return false;
    }

    // max iterations
    if (iter >= data_slice.max_iter) {
      printf(
          "pr_nibble::Stop_Condition: reached max iterations. breaking at "
          "it=%d\n",
          iter);
      return true;
    }

    // gradient too small
    if (!(*data_slice.h_grad_scale)) {
      printf(
          "pr_nibble::Stop_Condition: gradient too small. breaking at it=%d\n",
          iter);
      return true;
    }

    return false;
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
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

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
};  // end of PRNibbleIterationLoop

/**
 * @brief Template enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::GraphT::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT LabelT;
  typedef typename GraphT::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef PRNibbleIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief pr_nibble constructor
   */
  Enactor() : BaseEnactor("pr_nibble"), problem(NULL) {
    // <OPEN> change according to algorithmic needs
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
    // </OPEN>
  }

  /**
   * @brief pr_nibble destructor
   */
  virtual ~Enactor() { /*Release();*/
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
   * @brief Initialize the problem.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(
        problem, Enactor_None,
        // <OPEN> change to how many frontier queues, and their types
        2, NULL,
        // </OPEN>
        target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
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
   * @brief one run of pr_nibble, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        // <OPEN> change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 1,
        // </OPEN>
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(
      // <DONE> problem specific data if necessary, eg
      VertexT src, VertexT src_neib,
      // </DONE>
      util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    // <DONE> Initialize frontiers according to the algorithm:
    // In this case, we add a `src` + a neighbor to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 2;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 2 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForAll(
                [src, src_neib] __host__ __device__(VertexT * v,
                                                    const SizeT &i) {
                  v[i] = i == 0 ? src : src_neib;
                },
                2, target, 0));
          }
        }
      } else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }
    // </DONE>

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a pr_nibble computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(
      // <TODO> problem specific data if necessary, eg
      // VertexT src = 0
      // </TODO>
  ) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Template Done.", this->flag & Debug);
    return retval;
  }
};

}  // namespace pr_nibble
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
