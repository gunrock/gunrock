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
  // Display_Performance_Profiling(enactor);
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

/*
 * @brief Entry of gunrock_template function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the distances
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] distances  Return shortest distance to source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_louvain(gunrock::util::Parameters &parameters, GraphT &graph,
                       typename GraphT::VertexT *communities) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::louvain::Problem<GraphT> ProblemT;
  typedef gunrock::app::louvain::Enactor<ProblemT> EnactorT;
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
    problem.Reset(target);
    enactor.Reset(target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(communities, NULL, target);
  }

  enactor.Release(target);
  problem.Release(target);
  return total_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform Louvain
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[out] distances   Return shortest distance to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int,
          typename GValueT = unsigned int, typename TValueT = GValueT>
float louvain(const SizeT num_nodes, const SizeT num_edges,
              const SizeT *row_offsets, const VertexT *col_indices,
              const GValueT *edge_values, const int num_runs,
              VertexT *communities) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("louvain");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::louvain::UseParameters(parameters);
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
  graph.CsrT::edge_values.SetPointer((GValueT *)edge_values, num_edges,
                                     gunrock::util::HOST);
  // graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the Louvain
  double elapsed_time = gunrock_louvain(parameters, graph, communities);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

float louvain(const int num_nodes, const int num_edges, const int *row_offsets,
              const int *col_indices, const int *edge_values,
              int *communities) {
  return louvain(num_nodes, num_edges, row_offsets, col_indices, edge_values,
                 1 /* num_runs */, communities);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
