// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file knn_app.cu
 *
 * @brief Simple Gunrock Application
 */

// Gunrock api
#include <gunrock/gunrock.h>

// Test utils
#include <gunrock/util/test_utils.cuh>

// Graphio include
#include <gunrock/graphio/graphio.cuh>

// App and test base includes
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// JSON includes
#include <gunrock/util/info_rapidjson.cuh>

// KNN includes
#include <gunrock/app/knn/knn_enactor.cuh>
#include <gunrock/app/knn/knn_test.cuh>

namespace gunrock {
namespace app {
namespace knn {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "tag", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, "",
      "tag info for json string", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "k",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Numbers of k neighbors.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "x",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "Index of reference point.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "y",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "Index of reference point.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "cpu-elapsed", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0.0f,
      "CPU implementation, elapsed time (ms) for JSON.", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run knn tests
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
                     typename GraphT::SizeT k, typename GraphT::SizeT *h_knns,
                     typename GraphT::SizeT *ref_knns, util::Location target) {
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
  util::Info info("knn", parameters, graph);

  VertexT point_x = parameters.Get<int>("x");
  VertexT point_y = parameters.Get<int>("y");

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, k, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(point_x, point_y, k, target));
    GUARD_CU(enactor.Reset(target));

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
      GUARD_CU(problem.Extract(graph.nodes, k, h_knns));
      SizeT num_errors =
          Validate_Results(parameters, graph, h_knns, ref_knns, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(graph.nodes, k, h_knns));
  if (validation == "last") {
    SizeT num_errors =
        Validate_Results(parameters, graph, h_knns, ref_knns, false);
  }

  // compute running statistics
  // Change NULL to problem specific per-vertex visited marker, e.g.
  // h_distances
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
  // Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
