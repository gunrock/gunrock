// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gtc_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/gcn/graphsum/graphsum_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace graphsum {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of SSSP iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct GraphsumIterationLoop
    : public IterationLoopBase<EnactorT, Iteration_Default> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Iteration_Default>
      BaseIterationLoop;

  GraphsumIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of sssp, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data sssp that works on
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
    auto &dim = data_slice.dim;
    auto &in = data_slice.input;
    auto &out = data_slice.output;
    auto &local_vertices = data_slice.local_vertices;
    auto &forward = data_slice.forward;

    // The advance operation
    auto forward_lambda =
        [in, out, graph, dim] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      ValueT coef = (long long)graph.GetNeighborListLength(src) *
          graph.GetNeighborListLength(dest);
      coef = 1.0 / sqrt(coef);
      for (int i = 0; i < dim; i++)
        atomicAdd(out + src * dim + i, *(in + dest * dim + i) * coef);
      return true;
    };
    auto backward_lambda =
        [in, out, graph, dim] __host__ __device__(
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
      ValueT coef = (long long)graph.GetNeighborListLength(src) *
          graph.GetNeighborListLength(dest);
      coef = 1.0 / sqrt(coef);
      for (int i = 0; i < dim; i++)
        atomicAdd(out + src * dim + i, *(in + dest * dim + i) * coef);
      return true;
    };
    frontier.queue_length = local_vertices.GetSize();
    frontier.queue_reset = true;
    oprtr_parameters.advance_mode = data_slice.lb_mode;
    auto null_ptr = &local_vertices;
    null_ptr = NULL;
    if (forward)
      {
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V> (
            graph.csr (), &local_vertices, null_ptr, oprtr_parameters,
            forward_lambda));
      } else {
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V> (
            graph.csr (), &local_vertices, null_ptr, oprtr_parameters,
            backward_lambda));
      }

    enactor_stats.edges_queued[0] += graph.edges;
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

//    auto &data_slices = this->enactor->problem->data_slices;
//    bool all_zero = true;
//    for (int gpu = 0; gpu < num_gpus; gpu++)
//      if (data_slices[gpu]->num_updated_vertices)  // PR_queue_length > 0)
//      {
//        // printf("data_slice[%d].PR_queue_length = %d\n", gpu,
//        // data_slice[gpu]->PR_queue_length);
//        all_zero = false;
//      }
//    if (all_zero) return true;

    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (enactor_slices[gpu * num_gpus].enactor_stats.iteration < 1) {
        // printf("enactor_stats[%d].iteration = %lld\n", gpu, enactor_stats[gpu
        // * num_gpus].iteration);
        return false;
      }
    return true;
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
    return cudaSuccess;
  }
};  // end of SSSPIteration

/**
 * @brief SSSP enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
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
  typedef GraphsumIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief graphsumEnactor constructor
   */
  Enactor() : BaseEnactor("sssp"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief SSSPEnactor destructor
   */
  virtual ~Enactor() {
     Release();
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
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
        auto &frontier =
            this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
        // TODO: check whether the frontier is initialized properly
        frontier.queue_length =
            (peer_ != 0)
            ? 0
            : this->problem->data_slices[gpu]->local_vertices.GetSize();
        frontier.queue_index = 0;  // Work queue index
        frontier.queue_reset = true;
        this->enactor_slices[gpu * this->num_gpus + peer_]
            .enactor_stats.iteration = 0;
      }

    }
    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief one run of sssp, to be called within GunrockThread
   * @param thread_data Data for the CPU threadt
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 1, IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a SSSP computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU graphsum Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace grpahsum
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
