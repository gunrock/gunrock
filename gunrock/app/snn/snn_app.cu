// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file snn_app.cu
 *
 * @brief Simple Gunrock Application
 */

#include <cstdio>
#include <iostream>

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

// SNN includes
#include <gunrock/app/snn/snn_enactor.cuh>
#include <gunrock/app/snn/snn_test.cuh>

namespace gunrock {
namespace app {
namespace snn {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "labels-file",
      util::REQUIRED_ARGUMENT | util::REQUIRED_PARAMETER, 
      "", "List of points of dim-dimensional space", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "transpose",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "False if lables are not transpose", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "knn-version",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "gunrock", "Version of knn: \"gunrock\" or \"kmcuda\" or \"cuml\" or \"faiss\"", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "snn-tag", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, "",
      "snn-tag info for json string", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "n",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Numbers of points", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "dim",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Dimension of labels", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "k",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Numbers of k neighbors.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "eps",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "The minimum number of neighbors two points should share\n"
      "to be considered close to each other",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "min-pts",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "The minimum density that a point should have to be considered a core "
      "point\n",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "NUM-THREADS",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      128, "Number of threads running per block.", __FILE__, __LINE__));
 
  GUARD_CU(parameters.Use<bool>(
      "use-shared-mem",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "True if kernel must use shared memory.", __FILE__, __LINE__));
 
  GUARD_CU(parameters.Use<float>(
      "cpu-elapsed", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0.0f,
      "CPU implementation, elapsed time (ms) for JSON.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "knn-elapsed", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0.0f,
      "KNN Gunrock implementation, elapsed time (ms) for JSON.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "save-snn-results",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Save cluster assignments to file", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "snn-output-file",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "snn_results.output", "Filename of snn output", __FILE__, __LINE__));

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
template <typename GraphT, typename SizeT = typename GraphT::SizeT>
cudaError_t RunTests(
    util::Parameters &parameters, GraphT &graph, SizeT num_points, SizeT k,
    SizeT eps, SizeT min_pts,
    SizeT *h_knns,
    SizeT *h_cluster, SizeT *ref_cluster,
    SizeT *h_core_point_counter, SizeT *ref_core_point_counter,
    SizeT *h_noise_point_counter, SizeT *ref_noise_point_counter, 
    SizeT *h_cluster_counter, SizeT *ref_cluster_counter, 
    util::Location target) {
  cudaError_t retval = cudaSuccess;

  //typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  bool save_snn_results = parameters.Get<bool>("save-snn-results");
  std::string snn_output_file = parameters.Get<std::string>("snn-output-file");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("snn", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(h_knns, target));
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
      GUARD_CU(problem.Extract(num_points, k, h_cluster, h_core_point_counter,
                  h_noise_point_counter, h_cluster_counter));
      SizeT num_errors = Validate_Results(parameters, graph, h_cluster,
                               h_core_point_counter, h_noise_point_counter, 
                               h_cluster_counter,
                               ref_cluster, ref_core_point_counter, 
                               ref_noise_point_counter, 
                               ref_cluster_counter, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(num_points, k, h_cluster, h_core_point_counter, 
              h_noise_point_counter, h_cluster_counter));
  if (validation == "last") {
    SizeT num_errors = Validate_Results(parameters, graph, h_cluster,
                                h_core_point_counter, h_noise_point_counter, 
                                h_cluster_counter,
                                ref_cluster, ref_core_point_counter, 
                                ref_noise_point_counter, 
                                ref_cluster_counter, false);
  }

  // compute running statistics
  // Change NULL to problem specific per-vertex visited marker, e.g.
  // h_distances
  info.ComputeTraversalStats(enactor, (SizeT *)NULL);
  // Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // For JSON output
  info.SetVal("num-corepoints", std::to_string(h_core_point_counter[0]));
  info.SetVal("num-noisepoints", std::to_string(h_noise_point_counter[0]));
  info.SetVal("num-clusters", std::to_string(h_cluster_counter[0]));
  // info.SetVal("cpu-elapsed",
  // std::to_string(parameters.Get<float>("cpu-elapsed")));
  if (save_snn_results){
    std::ofstream output(snn_output_file);
    for (int i=0; i<num_points; ++i){
        output << h_cluster[i] << "\n";
    }
    output.close();
  }

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
