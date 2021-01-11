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
#include <gunrock/util/array_utils.cuh>

// Graphio include
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/graphio/labels.cuh>

// App and test base includes
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// KNN includes
#include <gunrock/app/knn/knn_helpers.cuh>
#include <gunrock/app/knn/knn_enactor.cuh>
#include <gunrock/app/knn/knn_test.cuh>

#include <iostream>
#include <algorithm>
#include <iterator>

//#define KNN_APP_DEBUG
#ifdef KNN_APP_DEBUG
    #define debug(a...) printf(a)
#else
    #define debug(a...)
#endif

namespace gunrock {
namespace app {
namespace knn {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  GUARD_CU(UseParameters_test(parameters));

  GUARD_CU(parameters.Use<int>(
      "n",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0, "Number of points in dim-dimensional space", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "labels-file",
      util::REQUIRED_ARGUMENT | util::REQUIRED_PARAMETER, 
      "", "List of points of dim-dimensional space", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "transpose",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "If false then labels will not transpose", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "dim",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      2, "Dimensions of space", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "k",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Number of k neighbors.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "NUM-THREADS",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      128, "Number of threads running per block.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "use-shared-mem",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "True if kernel must use shared memory.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "save-knn-results",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "If true then knn array will save to file.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "knn-output-file",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "knn_output", "File name.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "cpu-elapsed", 
      util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0.0f,
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
template <typename GraphT, typename ArrayT>
cudaError_t RunTests(util::Parameters &parameters,
        ArrayT& points,
        GraphT& graph,
        typename GraphT::SizeT n,
        typename GraphT::SizeT dim,
        typename GraphT::SizeT k,
        typename GraphT::SizeT *h_knns,
        typename GraphT::SizeT *ref_knns,
        util::Location target) {
  
  cudaError_t retval = cudaSuccess;

  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT; 

  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  bool save_knn_results = parameters.Get<bool>("save-knn-results");
  std::string knn_output_file = parameters.Get<std::string>("knn-output-file");
  util::Info info("knn", parameters, graph);

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
    GUARD_CU(problem.Reset(points.GetPointer(util::HOST), target));
    GUARD_CU(enactor.Reset(n, k, target));

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
      GUARD_CU(problem.Extract(h_knns));
#ifdef KNN_APP_DEBUG
      debug("extracted knns:\n");
      for (int i=0; i<n; ++i){
          debug("point %d\n", i);
          for (int j=0; j<k; ++j){
              debug("%d ", h_knns[i*k + j]);
          }
          debug("\n");
      }
#endif

      util::PrintMsg("-------------Validation-----------");
      SizeT num_errors =
          Validate_Results(parameters, graph, h_knns, ref_knns, points, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(h_knns));
#ifdef KNN_APP_DEBUG
      debug("extracted knns:\n");
      for (int i=0; i<n; ++i){
          debug("point %d\n", i);
          for (int j=0; j<k; ++j){
              debug("%d ", h_knns[i*k + j]);
          }
          debug("\n");
      }
#endif

  if (validation == "last") {
      util::PrintMsg("-------------Validation-----------");
    SizeT num_errors =
        Validate_Results(parameters, graph, h_knns, ref_knns, points, false);
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, (SizeT *)NULL);
  // Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(&enactor);
#endif

  if (save_knn_results){
      std::ofstream output(knn_output_file);
      for (int i=0; i<n-1; ++i){
          copy(h_knns + (k * i), h_knns + (k * (i+1)), std::ostream_iterator<ValueT>(output, " "));
          output << "\n";
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
