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
#include <gunrock/app/gcn/gcn_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#define PRINT_TIMER_AVERAGE(T, E) printf(#T " average time: %.3fms\n", T / E)

namespace gunrock {
namespace app {
namespace gcn {

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
struct GCNIterationLoop
    : public IterationLoopBase<EnactorT, Iteration_Default> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Iteration_Default>
      BaseIterationLoop;

  GCNIterationLoop() : BaseIterationLoop() {}

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
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;
    auto &modules = data_slice.modules;
    auto &vars = data_slice.vars;
    auto &eps = data_slice.eps, &learning_rate = data_slice.learning_rate, &beta1 = data_slice.beta1,
    &beta2 = data_slice.beta2, &weight_decay = data_slice.weight_decay;
    auto &training = data_slice.training;
    auto &w0 = data_slice.w0;
    auto &penalty = data_slice.penalty;
    auto &truth = data_slice.truth;
    auto &out = data_slice.AAxw0w1;
    auto &cnt = data_slice.cnt;
    auto &out_dim = data_slice.out_dim;
    auto &wrong = data_slice.wrong;
    auto &label = data_slice.label;
    auto &split = data_slice.split;
    auto &in_feature = data_slice.in_feature;
    auto &timer = data_slice.timer;

    timer.Start();

    // in_feature is the size of num edges
    // in_feature is populated by subgraph CSR edgevalues
    GUARD_CU (data_slice.x_val.ForEach(in_feature,
        []__host__ __device__(ValueT &dst, ValueT &src) {
      dst = src;
    }, in_feature.GetSize(), util::DEVICE))

    // split is training testing split
    // this split is populated by a split file and has a 
    // boolean for each node
    
    GUARD_CU(truth.ForAll([label, split]__host__ __device__(int *t, SizeT &i) {
      // if split is 1 then node is for training else if the value is 2
      // node is for testing
      t[i] = split[i] == 1 ? label[i] : -1;
    }))

    // modules added by user
    // the ordering matters
    // how to add pooling as a module
    // and what is the invariant of a module
    // what happens if the graph changes?

    for (auto m : modules) {
      // bool is for train vs test
      // on train the backpropagation values need not be stored

      // shouldn't the training flag be used here?
      m->forward(true);
    }

//    GUARD_CU (data_slice..Print())

    std::pair<double, double> pair;

    if (training) {
      ValueT step_size = learning_rate * sqrt(1 - pow (beta2, iteration + 1)) / (1 - pow (beta1, iteration + 1));
      double train_loss, train_acc, val_loss, val_acc;
      // the loss from the cross entropy is computed here
      get_loss_acc(data_slice, pair);
      std::tie(train_loss, train_acc) = pair;
      for (int i = modules.size() - 1; i >= 0; i--) {
        // how do the modules run backprop independently without the corresponding errors from the layer in front?
        // the oss (grad) is shared using common Array1D
        // for example between the graph sum and the cross entropy
        // the AAxW_grad arrays are shared
        // this is how its done
        
        modules[i]->backward();
      }

      // seems like vars has two entries

      // corresponding to w0 and w1 respectively
      
      // the weights get modified here
      // the gradients are stored during the modules backward()
      
      for (auto var : vars) {
        auto &w = var.weights, &g = var.grads, &m = var.m, &v = var.v;
        auto &decay = var.decay;
        // TODO: step_size may need to be calculated on the fly
        GUARD_CU (w.ForAll(
            [decay, g, m, v, weight_decay, beta1, beta2, step_size, eps]__host__ __device__(ValueT *ws, SizeT &i) {
              ValueT grad = g[i];
              if (decay) grad += weight_decay * ws[i];
              m[i] = beta1 * m[i] + (1 - beta1) * grad;
              v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
              ws[i] -= step_size * m[i] / (sqrt(v[i]) + eps);
        }))
//        w.Print("w: ", 10, util::DEVICE);
//        g.Print("g: ", 10, util::DEVICE);
//        m.Print("m: ", 10, util::DEVICE);
//        v.Print("v: ", 10, util::DEVICE);
      }

      // 
      GUARD_CU (data_slice.x_val.ForEach(in_feature,
          []__host__ __device__(ValueT &dst, ValueT &src) {
        dst = src;
      }, in_feature.GetSize(), util::DEVICE))

      GUARD_CU(truth.ForAll([label, split]__host__ __device__(int *t, SizeT &i) {
        t[i] = split[i] == 2 ? label[i] : -1;
      }))
      for (auto m : modules) {
        m->forward(false);
      }
      get_loss_acc(data_slice, pair);
      // validation loss and accuracy
      // for all nodes that have the split as 2
      // their forwards do not compute gradients
      std::tie(val_loss, val_acc) = pair;
      timer.Stop();
      printf("epoch: %d, train_loss: %lf, train_acc: %lf, val_loss: %lf, val_acc: %lf, time = %fms\n",
          iteration + 1, train_loss, train_acc, val_loss, val_acc, timer.ElapsedMillis());
      data_slice.tot_time += timer.ElapsedMillis();
    }

    // final prediction
    //  
    if (iteration == data_slice.max_iter - 1) {
        GUARD_CU (data_slice.x_val.ForEach(in_feature,
            []__host__ __device__(ValueT &dst, ValueT &src) {
          dst = src;
        }, in_feature.GetSize(), util::DEVICE))
        GUARD_CU(truth.ForAll([label, split]__host__ __device__(int *t, SizeT &i) {
          t[i] = split[i] == 1 ? label[i] : -1;
        }))

        for (auto m : modules) {
          m->forward(false);
        }
        get_loss_acc(data_slice, pair);
        printf("test_loss: %lf, test_acc: %lf, avg_train_time: %fms\n",
            pair.first, pair.second, data_slice.tot_time / data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.fw_sprmul, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.bw_sprmul, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.fw_graphsum, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.bw_graphsum, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.fw_dropout, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.bw_dropout, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.fw_relu, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.bw_relu, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.fw_matmul, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.bw_matmul, data_slice.max_iter);
        PRINT_TIMER_AVERAGE(data_slice.fw_loss, data_slice.max_iter);
    }
    return retval;
  }

  template <typename T>
  cudaError_t get_loss_acc(T &data_slice, std::pair<double, double> &res) {
    auto retval = cudaSuccess;
    auto &penalty = data_slice.penalty;
    auto &weight_decay = data_slice.weight_decay;
    auto &wrong = data_slice.wrong;
    auto &out_dim = data_slice.out_dim;
    auto &cnt = data_slice.cnt;
    auto &out = data_slice.AAxw0w1;
    auto &w0 = data_slice.w0;
    auto &truth = data_slice.truth;
    auto &modules = data_slice.modules;


    // computes the regularised loss from w0 weights
    // why is w1 not being used for the loss regularisation
    // penalty is an array of size 1 which stores it 
    GUARD_CU (penalty.ForEach([]__host__ __device__(ValueT &x) { x = 0; }))
    GUARD_CU (w0.ForEach(
              [penalty]__host__ __device__(ValueT &x) {
      atomicAdd (penalty + 0, x * x);
    }))
    GUARD_CU (penalty.ForEach([weight_decay]__host__ __device__(ValueT &x) {
      x *= weight_decay / 2;
    }))
//      GUARD_CU(penalty.Print())
    ValueT loss;

    // move the loss to the cpu
    GUARD_CU (penalty.SetPointer(&loss, 1, util::HOST))
    GUARD_CU (penalty.Move(util::DEVICE, util::HOST))
    
    // get the loss from the last module
    // in our case its the cross entropy loss

    loss += modules.back()->GetLoss();
//      std::cout << "loss: " << modules.back()->GetLoss() << ", ";

    GUARD_CU (cnt.ForEach([]__host__ __device__(int &x) { x = 0; }))
    GUARD_CU (wrong.ForEach([]__host__ __device__(int &x) { x = 0; }))
//      GUARD_CU(truth.Print())
    GUARD_CU (truth.ForAll(
        [wrong, out_dim, cnt, out]__host__ __device__(int *label, SizeT &i) {
      if (label[i] < 0) return;
      atomicAdd (cnt + 0, 1);
      auto logit = out + i * out_dim;
      for (int j = 0; j < out_dim; j++) {
          if (logit[j] > logit[label[i]]) {
              atomicAdd (wrong + 0, 1);
              return;
            }
        }
    }, truth.GetSize(), util::DEVICE))

    int total, wrong_h;
    GUARD_CU (cnt.SetPointer(&total, 1, util::HOST))
    GUARD_CU (wrong.SetPointer(&wrong_h, 1, util::HOST))
    GUARD_CU (cnt.Move(util::DEVICE, util::HOST))
    GUARD_CU (wrong.Move(util::DEVICE, util::HOST))
    ValueT acc = (total - wrong_h) * 1.0 / total;
    res = std::make_pair(loss, acc);
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
      if (enactor_slices[gpu * num_gpus].enactor_stats.iteration < data_slices[0]->max_iter) {
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
  typedef GCNIterationLoop<EnactorT> IterationT;

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
    util::PrintMsg("GPU gcn Done.", this->flag & Debug);
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
