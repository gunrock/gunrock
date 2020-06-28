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
#include <gunrock/app/gcn/CrossEntropyLoss/CrossEntropyLoss_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace CrossEntropyLoss {

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
struct CorssEntropyLoop
    : public IterationLoopBase<EnactorT, Iteration_Default> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Iteration_Default>
      BaseIterationLoop;

  CorssEntropyLoop() : BaseIterationLoop() {}

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
//    auto &graph = data_slice.sub_graph[0];
//    auto &frontier = enactor_slice.frontier;
//    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
//    auto &iteration = enactor_stats.iteration;
    auto &ground_truth = data_slice.ground_truth;
    auto &loss = data_slice.loss;
    auto &logits = data_slice.logits;
    auto &n_nodes = data_slice.num_nodes, &num_clases = data_slice.num_classes;
    auto &grad = data_slice.grad;
    auto &training = data_slice.training;

    util::Array1D<SizeT, int> count("count");
    GUARD_CU(count.Init(1, util::DEVICE))
    GUARD_CU(count.ForEach([]__host__ __device__(int &x) { x = 0; }))

    util::Array1D<SizeT, ValueT> max_logits("max_logits");
    GUARD_CU(max_logits.Init(data_slice.num_nodes, util::DEVICE))
//    GUARD_CU (ground_truth.Print())
    GUARD_CU(ground_truth.ForAll(logits, grad,
        [max_logits, num_clases, loss, training, count]__host__ __device__(int *truth,
            ValueT *d_logits, ValueT *grad, const SizeT &pos) {
          if (truth[pos] >= 0) {
            // count the number of labeled nodes
            atomicAdd(count + 0, 1);

            // get max_logit for current node, max_logits will be used for
            // normalization trick: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/?nsukey=rWH8DtvqNfw72DAmoQPHjdYV2Yywr0HdCvKkgA4vR0X4xQOt5X2VyNstwytbKOI8CP7Mhsa84C0WqsVrgIPNPjOBqayTXF8ufC76oms71y2l9aq2ojHp4NeSPqnweprhc7IQ1rBBsPpdYPEBe2hEO33xb0XMT5J%2F2TzpcyFIw8GU5dPDtoqDzW%2BGOcWLHbvPxBBrYioidJYAS3TMuinEXQ%3D%3D
            ValueT sum_exp = 0, *logit = d_logits + pos * num_clases;
            ValueT max_logit = util::PreDefinedValues<ValueT>::MinValue;
            for (int i = 0; i < num_clases; i++) {
              max_logit = max(max_logit, logit[i]);
            }
            max_logits[pos] = max_logit;

            // get sum_exp for calculating sigmoid function
            for (int i = 0; i < num_clases; i++) {
              logit[i] -= max_logit;
              sum_exp += exp(logit[i]);
            }

            atomicAdd(loss + 0, log(sum_exp) - logit[truth[pos]]);

            if (training) {
              for (int i = 0; i < num_clases; i++) {
                ValueT prob = exp(logit[i]) / sum_exp;
                grad[pos * num_clases + i] = prob;
              }
            }
            grad[pos * num_clases + truth[pos]] -= 1;
          }
        }, n_nodes, util::DEVICE
    ))
//    max_logits.Print();

//    loss.Print();
//    count.Print();
    GUARD_CU(loss.ForEach(
        count,
        []__host__ __device__(ValueT &l, int &c) {
          l /= c;
        }, 1, util::DEVICE
    ))
    if (training) {
      GUARD_CU(grad.ForEach(
          [count]__host__ __device__(ValueT &x) {
            x /= count[0];
          }, grad.GetSize(), util::DEVICE
      ))
    }

//    GUARD_CU(loss.Print())

//    grad.Print();
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

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      auto &enactor_stats = enactor_slices[gpu * num_gpus].enactor_stats;
      if (enactor_stats.iteration < 1) {
//         printf("enactor_stats[%d].iteration = %lld\n", gpu, enactor_stats.iteration);
        return false;
      }
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
  typedef CorssEntropyLoop<EnactorT> IterationT;

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
//      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
//      auto &graph = problem.sub_graphs[gpu];
//      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
//                                               this->queue_factors));
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
        frontier.queue_length = (peer_ != 0) ? 0 : 1;
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
    util::PrintMsg("GPU cross entropy loss calculation Done.", this->flag & Debug);
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
