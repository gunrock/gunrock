// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kcore_enactor.cuh
 *
 * @brief K-Core Problem Enactor
 */

#pragma once

#include <gunrock/util/sort_device.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>

#include <gunrock/app/kcore/kcore_problem.cuh>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#include <gunrock/oprtr/1D_oprtr/for.cuh>
#endif

#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace kcore {

/**
 * @brief Speciflying parameters for K-Core Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of K-Core iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct KCoreIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  KCoreIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of k-core, one iteration
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

    auto &num_cores = data_slice.num_cores;
    auto &out_degrees = data_slice.out_degrees;
    auto stream = oprtr_parameters.stream;
    util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    auto null_ptr = null_frontier;

    // TODO: K-Core implementation
    // Can Gunrock:
    // 1) do filter before advance? (no technical difficulty)
    // 2) take one input frontier, output two frontiers?
    // 3) specify frontier to operate on and take attribute arbitrarily.

    auto deg_less_than_k_op =
          [k, out_degrees, num_cores] __host__ __device__ (
              const vertext &src, vertext &dest, const sizet &edge_id,
              const vertext &input_item, const sizet &input_pos,
              sizet &output_pos) -> bool {
                if (out_degrees[src] < k) {
                  num_cores[src] = k - 1;
                  out_degrees[src] = 0;
                  return true;
                } else {
                  return false;
                }
          };
        
    auto deg_at_least_k_op =
          [k, out_degrees] __host__ __device__ (
              const vertext &src, vertext &dest, const sizet &edge_id,
              const vertext &input_item, const sizet &input_pos,
              sizet &output_pos) -> bool {
                return (out_degrees[src] >= k);
          };
      
    auto update_deg_op =
          [k, out_degrees] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
                     const VertexT &input_item, const SizeT &input_pos,
                     SizeT &output_pos) -> bool {
                 atomicAdd(out_degrees[dest], -1);
                 return out_degrees[dest] > 0;      
          };
    for (int k = 1; k <= graph.nodes; ++k) {
      // set frontier to all nodes (null_frontier means all)
      frontier = null_ptr;
      while (true) {
        // filter(input, to_remove)
        GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier == null_ptr ? null_ptr : frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          deg_less_than_k_op));
        
        GUARD_CU(frontier.work_progress.GetQueueLength(
          frontier.queue_index, frontier.queue_length, false,
          oprtr_parameters.stream, true));

        // if (to_remove.len() == 0) {
        if (frontier.queue_length == 0) {
          // reset frontier
          frontier.Next_V_Q();
          // filter(input, remaining);
          GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          deg_at_least_k_op));
          
          break;
        } else {
          // advance(to_remove, empty_q);
          GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), null_frontier, oprtr_parameters,
          advance_op));
          // reset frontier
          frontier.Next_V_Q();
          // filter(input, remaining);
          GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
          graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), oprtr_parameters,
          deg_at_least_k_op));
        }
      }
      // if (remaining.len() == 0) break;
      GUARD_CU(frontier.work_progress.GetQueueLength(
          frontier.queue_index, frontier.queue_length, false,
          oprtr_parameters.stream, true));
      if (frontier.queue_length == 0) break;
    }

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto num_remaining_nodes = data_slice->num_remaining_nodes;

    if (num_remaining_nodes == 0) {
      return true;
    } else {
      return false;
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
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of kcoreIteration

/**
 * @brief K-Core enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor : public EnactorBase<typename _Problem::GraphT,
                                   typename _Problem::GraphT::VertexT,
                                   typename _Problem::GraphT::ValueT,
                                   ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef EnactorBase<GraphT, VertexT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef KCoreIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief kcore constructor
   */
  Enactor() : BaseEnactor("KCore"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief kcore destructor
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
   * @brief one run of kcore, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        //       per element in the inter-GPU sub-frontiers
        0, 1, IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));

    SizeT num_nodes = this->problem->data_slices[0][0].sub_graph[0].nodes;

    // In this case, we add a single `src` to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = num_nodes;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? num_nodes : 0;
          if (peer_ == 0) {
            util::Array1D<SizeT, VertexT> tmp;
            tmp.Allocate(num_nodes, target | util::HOST);
            for (SizeT i = 0; i < num_nodes; ++i) {
              tmp[i] = (VertexT)i % num_nodes;
            }
            GUARD_CU(tmp.Move(util::HOST, target));

            GUARD_CU(frontier.V_Q()->ForEach(
                tmp,
                [] __host__ __device__(VertexT & v, VertexT & i) { v = i; },
                num_nodes, target, 0));

            tmp.Release();
          }
        }
      } else {
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a kcore computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU K-Core Done.", this->flag & Debug);
    return retval;
  }
};

}  // namespace kcore
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
