// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file louvain_app.cu
 *
 * @brief Community Detection (Louvain) application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/louvain/louvain_enactor.cuh>
#include <gunrock/app/louvain/louvain_test.cuh>

namespace gunrock {
namespace app {
namespace louvain {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  GUARD_CU(UseParameters_test(parameters));
  return retval;
}

/**
 * @brief Run Louvain tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_distances Reference distances
 * @param[in]  target        Whether to perform the Louvain
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::VertexT *ref_communities = NULL,
                     util::Location target = util::DEVICE) {
  cudaError_t retval = cudaSuccess;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;
  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // parse configurations from parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("Louvain", parameters, graph);  // initialize Info structure

  // allocate problem specific host data, e.g.:
  VertexT *h_communities = new VertexT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  // perform the algorithm
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(target));
    GUARD_CU(enactor.Reset(target));
    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg("--------------------------", !quiet_mode);

    if (validation == "each") {
      util::PrintMsg(
          "Run " + std::to_string(run_num) + " elapsed: " +
              std::to_string(cpu_timer.ElapsedMillis()) + " ms, #passes = " +
              std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
          !quiet_mode);

      GUARD_CU(problem.Extract(h_communities, NULL, target));
      SizeT num_errors = app::louvain::Validate_Results(
          parameters, graph, h_communities, ref_communities);
    } else {
      util::PrintMsg(
          "Run " + std::to_string(run_num) + " elapsed: " +
              std::to_string(cpu_timer.ElapsedMillis()) + " ms, #passes = " +
              std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
          !quiet_mode);
    }
  }

  cpu_timer.Start();
  // Copy out results
  GUARD_CU(problem.Extract(h_communities, NULL, target));
  if (validation == "last") {
    SizeT num_errors = app::louvain::Validate_Results(
        parameters, graph, h_communities, ref_communities);
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
  delete[] h_communities;
  h_communities = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace louvain
}  // namespace app
}  // namespace gunrock


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
