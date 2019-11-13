// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file kcore_app.cu
 *
 * @brief K-Core Gunrock Application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definitions
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>
#include <gunrock/graphio/graphio.cuh>

// K-Core
#include <gunrock/app/kcore/kcore_enactor.cuh>
#include <gunrock/app/kcore/kcore_test.cuh>

// Others
#include <cstdio>

namespace gunrock {
namespace app {
namespace kcore {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  GUARD_CU(UseParameters_test(parameters));

  return retval;
}

/**
 * @brief Run kcore tests
 * @tparam     GraphT        Type of the graph
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  target        where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::VertexT *ref_num_cores,
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

  VertexT *h_num_cores = new VertexT[graph.nodes];
  VertexT max_k = 0;

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(graph, target));
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
      GUARD_CU(problem.Extract(h_num_cores));
      SizeT num_errors = Validate_Results(parameters, graph, h_num_cores,
                                          ref_num_cores, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(h_num_cores));
  if (validation == "last") {
    SizeT num_errors = Validate_Results(parameters, graph, h_num_cores, ref_num_cores,
                                        false);
  }

  // Print Max K-Core
  for (SizeT v = 0; v < graph.nodes; v++) {
    int k = h_num_cores[v];
    if (k > max_k) {
      max_k = k;
    }
  }

  util::PrintMsg("Max K-Core: " + std::to_string(max_k), !quiet_mode);

  // compute running statistics
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(&enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_num_cores;
  h_num_cores = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace kcore
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_kcore function
 * @tparam     GraphT     Type of the graph
 * @tparam     VertexT    Type of the num_cores
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] num_cores  Return generated core number for each run
 * @param[out] max_k      Return max K-Core generated for each run
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename VertexT = typename GraphT::VertexT,
          typename SizeT = typename GraphT::SizeT>
double gunrock_kcore(gunrock::util::Parameters &parameters, GraphT &graph,
                     VertexT **num_cores, VertexT *max_k) {
  typedef gunrock::app::kcore::Problem<GraphT> ProblemT;
  typedef gunrock::app::kcore::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, target);
  enactor.Init(problem, target);

  int num_runs = parameters.Get<int>("num-runs");
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    problem.Reset(graph, target);
    enactor.Reset(target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(num_cores[run_num]);

    // find max k
    for (SizeT v = 0; v < graph.nodes; v++) {
      int k = num_cores[run_num][v];
      if (k > max_k[run_num]) {
        max_k[run_num] = k;
      }
    }
  }

  enactor.Release(target);
  problem.Release(target);
  return total_time;
}

/*
 * @brief Entry of gunrock_kcore function
 * @tparam     VertexT    Type of the k-core
 * @tparam     SizeT      Type of the num_cores
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] num_cores  Return generated core number for each run
 * @param[out] max_k      Return max K-Core generated for each run
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int,
          typename GValueT = unsigned int>
double kcore(const SizeT num_nodes, const SizeT num_edges,
             const SizeT *row_offsets, const VertexT *col_indices,
             const int num_runs, int **num_cores, int *max_k,
             const GValueT edge_values = NULL) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("kcore");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::kcore::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("num-runs", num_runs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer((SizeT *)row_offsets, num_nodes + 1,
                                     gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer((VertexT *)col_indices, num_edges,
                                        gunrock::util::HOST);
  // graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the K-Core
  double elapsed_time = gunrock_kcore(parameters, graph, num_cores, max_k);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

/*
 * @brief Entry of gunrock_kcore function
 * @tparam     VertexT    Type of the k-core
 * @tparam     SizeT      Type of the num_cores
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] num_cores  Return generated core number for each run
 * @param[out] max_k      Return max K-Core generated for each run
 * \return     double     Return accumulated elapsed times for all runs
 */
double kcore(const int num_nodes, const int num_edges, const int *row_offsets,
             const int *col_indices, int *num_cores, int max_k) {
  return kcore(num_nodes, num_edges, row_offsets, col_indices, 1 /* num_runs */,
               &num_cores, &max_k);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: