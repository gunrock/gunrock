// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file hits_app.cu
 *
 * @brief HITS Gunrock Application
 */

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

#include <gunrock/app/hits/hits_enactor.cuh>
#include <gunrock/app/hits/hits_test.cuh>

namespace gunrock {
namespace app {
namespace hits {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief Run hits tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  target        where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::ValueT *ref_hrank,
                     typename GraphT::ValueT *ref_arank,
                     util::Location target) {
  cudaError_t retval = cudaSuccess;

  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  bool quick_mode = parameters.Get<bool>("quick");
  int num_runs = parameters.Get<int>("num-runs");
  double tol = parameters.Get<double>("tol");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("HITS", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem specific host data
  ValueT *h_hrank = new ValueT[graph.nodes];
  ValueT *h_arank = new ValueT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(target));
    GUARD_CU(enactor.Reset(graph.nodes, target));

    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            ", #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);

    if (validation == "each") {
      GUARD_CU(problem.Extract(h_hrank, h_arank));
      SizeT num_errors = Validate_Results(parameters, graph, h_hrank, h_arank,
                                          ref_hrank, ref_arank, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(h_hrank, h_arank));

  if (validation == "last") {
    SizeT num_errors = Validate_Results(parameters, graph, h_hrank, h_arank,
                                        ref_hrank, ref_arank, tol, false);

    // num_errors stores how many positions are mismatched
    // Makes sense to keep this? Would need to sort first.
    if (!quiet_mode) {
      if (!quick_mode) {
        printf("CPU Algorithm Results:\n");
        DisplaySolution<GraphT>(ref_hrank, ref_arank, graph.nodes);
        printf("\n");
      }

      printf("GPU Algorithm Results:\n");
      DisplaySolution<GraphT>(h_hrank, h_arank, graph.nodes);
    }
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
#ifdef ENABLE_PERFORMANCE_PROFILING
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  // Release problem specific data, e.g.:
  delete[] h_hrank;
  h_hrank = NULL;
  delete[] h_arank;
  h_arank = NULL;

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace hits
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
