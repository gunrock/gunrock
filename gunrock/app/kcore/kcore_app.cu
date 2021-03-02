// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file kcore_app.cu
 *
 * @brief k-core vertex decomposition application
 */

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

#include <gunrock/app/kcore/kcore_enactor.cuh>
#include <gunrock/app/kcore/kcore_test.cuh>

namespace gunrock {
namespace app {
namespace kcore {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief Run k-core tests
 * @tparam     GraphT        Type of the graph
 * @param[in]  parameters    Execution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_k_cores   Reference k-core values
 * @param[in]  target        Where to perform the k-core compute
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::SizeT *ref_k_cores,
                     util::Location target) {
  cudaError_t retval = cudaSuccess;

  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("kcore", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  SizeT *h_k_cores = new SizeT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, gunrock::util::HOST, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(graph, target));
    GUARD_CU(enactor.Reset(target));

//    util::PrintMsg("__________________________", !quiet_mode);
    util::PrintMsg("__________________________", true);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            ", #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        true);

    if (validation == "each") {
      GUARD_CU(problem.Extract(h_k_cores));
      SizeT num_errors = Validate_Results(parameters, graph, h_k_cores, ref_k_cores, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(h_k_cores));
  if (validation == "last") {
    SizeT num_errors = Validate_Results(parameters, graph, h_k_cores, ref_k_cores, false);
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(&enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_k_cores;
  h_k_cores = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace kcore
}  // namespace app
}  // namespace gunrock
