// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file graphsum_app.cu
 *
 * @brief gcn graphsum application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/gcn/CrossEntropyLoss/CrossEntropyLoss_enactor.cuh>
#include <gunrock/app/gcn/CrossEntropyLoss/CrossEntropyLoss_test.cuh>

/**
 * @brief      graphsum layer of GCN
 *
 * @param      parameters  The parameters
 * @param      graph       The graph
 * @param[in]  dim         dimension of the feature vector
 * @param      in          the input to the graphsum layer
 * @param      out         output matrix
 *
 * @tparam     GraphT      type of the graph
 * @tparam     ValueT      type of the value, double by default
 *
 * @return     time elapsed to execute
 */

namespace gunrock {
namespace app {
namespace CrossEntropyLoss {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<int>(
      "num_classes",
      util::OPTIONAL_PARAMETER | util::SINGLE_VALUE | util::OPTIONAL_ARGUMENT,
      10, "number of classes per node", __FILE__, __LINE__
  ));

  GUARD_CU(parameters.Use<int>(
      "num_nodes",
      util::OPTIONAL_PARAMETER | util::SINGLE_VALUE | util::OPTIONAL_ARGUMENT,
      1000, "number of nodes", __FILE__, __LINE__
  ));

  return retval;
}

}
}
}

using namespace gunrock;

template <typename SizeT, typename ValueT, typename GraphT>
struct cross_entropy : module {
  typedef gunrock::app::CrossEntropyLoss::Problem<GraphT> ProblemT;
  typedef gunrock::app::CrossEntropyLoss::Enactor<ProblemT> EnactorT;
  typedef util::Array1D<SizeT, ValueT> Array;

  GraphT dummy;
  util::Array1D<SizeT, ValueT> logits, grad;
  util::Array1D<SizeT, int> truth;
  ProblemT *problem;
  EnactorT *enactor;
  int dim;
  float *fw_time;

  cross_entropy(util::Parameters &p, Array _logits, Array _grad,
                util::Array1D<SizeT, int> _truth, int num_nodes, int num_classes, float *_fw,
                bool training=true) :
          logits(_logits), grad(_grad), truth(_truth), fw_time(_fw) {
    problem = new ProblemT(p);
    enactor = new EnactorT();

    problem->Init(dummy, num_nodes, num_classes, logits, grad, truth, training);
    enactor->Init(*problem);
  }

  virtual void forward(bool train) override {
    timer.Start ();

    problem->Reset(train);
    enactor->Reset();
    enactor->Enact();

    timer.Stop ();
    *fw_time += timer.ElapsedMillis ();
  }

  virtual void backward() override {}

  virtual double GetLoss() override {
    double loss;
    problem->Extract (&loss);
    return loss;
  }
};

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CrossEntropyLoss(gunrock::util::Parameters &parameters, GraphT &graph, const int num_nodes,
                        const int num_classes, ValueT *logits, int *ground_truth, ValueT *grad, ValueT &loss) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::CrossEntropyLoss::Problem<GraphT> ProblemT;
  typedef gunrock::app::CrossEntropyLoss::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
//  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, num_nodes, num_classes, logits, ground_truth);
  enactor.Init(problem, target);

  problem.Reset();
  enactor.Reset();

  cpu_timer.Start();
  enactor.Enact();
  cpu_timer.Stop();

  total_time += cpu_timer.ElapsedMillis();
  problem.Extract(grad, &loss);

  enactor.Release(target);
  problem.Release(target);

  return total_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
