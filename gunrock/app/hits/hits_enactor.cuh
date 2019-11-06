// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_enactor.cuh
 *
 * @brief hits Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/reduce_device.cuh>

#include <gunrock/app/hits/hits_problem.cuh>

namespace gunrock {
namespace app {
namespace hits {

/**
 * @brief Speciflying parameters for hits Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of hits iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct hitsIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  hitsIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of hits, one iteration
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

    cudaStream_t stream = enactor_slice.stream;

    // HITS-specific data slices
    auto &hrank_curr = data_slice.hrank_curr;
    auto &arank_curr = data_slice.arank_curr;
    auto &hrank_next = data_slice.hrank_next;
    auto &arank_next = data_slice.arank_next;

    auto &cub_temp_space = data_slice.cub_temp_space;
    auto &hrank_mag = data_slice.hrank_mag;
    auto &arank_mag = data_slice.arank_mag;

    // Set the frontier to NULL to specify that it should include
    // all vertices
    util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    frontier.queue_length = graph.nodes;
    frontier.queue_reset = true;

    // Number of times to iterate the HITS algorithm
    auto max_iter = data_slice.max_iter;

    // Normalize the HITS scores every N iterations.
    // Provides speedup at the risk of data overflow
    auto normalize_n = data_slice.normalize_n;

    // Reset next ranks to zero
    auto reset_zero_op = [hrank_next, arank_next] __host__ __device__(
                             VertexT * v_q, const SizeT &pos) {
      hrank_next[pos] = (ValueT)0.0;
      arank_next[pos] = (ValueT)0.0;
    };

    GUARD_CU(frontier.V_Q()->ForAll(reset_zero_op, graph.nodes));

    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize Failed");

    // Advance operation to update all hub and auth scores
    auto advance_op =
        [hrank_curr, arank_curr, hrank_next, arank_next] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      // Update the hub and authority scores.
      // Look into NeighborReduce for speed improvements
      atomicAdd(&hrank_next[src], arank_curr[dest]);
      atomicAdd(&arank_next[dest], hrank_curr[src]);

      return true;
    };

    // Perform advance operation
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
        graph.csr(), null_frontier, null_frontier, oprtr_parameters,
        advance_op));

    GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize Failed");

    // After updating the scores, normalize the hub and array scores
    // either after every N iterations (user-specified, default=1),
    // or at the last iteration
    if (((iteration + 1) % normalize_n == 0) || iteration == (max_iter - 1)) {
      // 1) Square each element
      auto square_op = [hrank_next, arank_next] __host__ __device__(
                           VertexT * v_q, const SizeT &pos) {
        hrank_next[pos] = hrank_next[pos] * hrank_next[pos];
        arank_next[pos] = arank_next[pos] * arank_next[pos];
      };

      GUARD_CU(frontier.V_Q()->ForAll(square_op, graph.nodes));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize Failed");

      // 2) Sum all squared scores in each array
      GUARD_CU(util::cubReduce(
          cub_temp_space, hrank_next, hrank_mag, graph.nodes,
          [] __host__ __device__(const ValueT &a, const ValueT &b) {
            return a + b;
          },
          ValueT(0), stream));

      GUARD_CU(util::cubReduce(
          cub_temp_space, arank_next, arank_mag, graph.nodes,
          [] __host__ __device__(const ValueT &a, const ValueT &b) {
            return a + b;
          },
          ValueT(0), stream));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize Failed");

      auto normalize_divide_op =
          [hrank_next, arank_next, hrank_mag, arank_mag] __host__ __device__(
              VertexT * v_q, const SizeT &pos) {
            if (hrank_mag[0] > 0) {
              hrank_next[pos] = sqrt(hrank_next[pos]) / sqrt(hrank_mag[0]);
            }

            if (arank_mag[0] > 0) {
              arank_next[pos] = sqrt(arank_next[pos]) / sqrt(arank_mag[0]);
            }
          };
      // Divide all elements by the square root of their squared sums.
      // Note: take sqrt of x in denominator because x^2 was done in place.
      GUARD_CU(frontier.V_Q()->ForAll(normalize_divide_op, graph.nodes));

      GUARD_CU2(cudaStreamSynchronize(stream), "cudaStreamSynchronize Failed");
    }
    // After normalization, swap the next and current vectors
    auto hrank_temp = hrank_curr;
    hrank_curr = hrank_next;
    hrank_next = hrank_temp;

    auto arank_temp = arank_curr;
    arank_curr = arank_next;
    arank_next = arank_temp;

    // Possibly normalize only at the end, or every n iterations
    // for potential speed improvements. Additionally, look into
    // NeighborReduce for adding host and auth scores

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slices = this->enactor->enactor_slices;
    auto iter = enactor_slices[0].enactor_stats.iteration;
    auto user_iter = data_slice.max_iter;

    // user defined stop condition
    if (iter == user_iter) return true;
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

    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool { return true; };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of hitsIteration

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
  typedef hitsIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief hits constructor
   */
  Enactor() : BaseEnactor("hits"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief hits destructor
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
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL, target, false));
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
   * @brief one run of hits, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 1, IterationT>(
        thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
   * @param[in] nodes Number of nodes in the graph
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(typename GraphT::SizeT nodes,
                    util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? nodes : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForAll(
                [] __host__ __device__(VertexT * v_q, const SizeT &pos) {
                  v_q[pos] = pos;
                },
                nodes, target, 0));
          }
        }
      } else {  // Incomplete/untested
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
   * @brief Enacts a hits computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Template Done.", this->flag & Debug);
    return retval;
  }
};

}  // namespace hits
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
