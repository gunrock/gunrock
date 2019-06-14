// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file bfs_app.cu
 *
 * @brief Gunrock breadth-first search (BFS) application
 */

#include <gunrock/app/app.cuh>

// breadth-first search includes
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_test.cuh>

namespace gunrock {
namespace app {
namespace bfs {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "src",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "0",
      "<Vertex-ID|random|largestdegree> The source vertices\n"
      "\tIf random, randomly select non-zero degree vertices;\n"
      "\tIf largestdegree, select vertices with largest degrees",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "src-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "seed to generate random sources", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run BFS tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_labels    Reference labels
 * @param[in]  target        Whether to perform the BFS
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename LabelT = typename GraphT::VertexT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     LabelT **ref_labels = NULL,
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
  bool mark_pred = parameters.Get<bool>("mark-pred");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_srcs = srcs.size();
  util::Info info("BFS", parameters, graph);  // initialize Info structure

  // Allocate host-side array (for both reference and GPU-computed results)
  LabelT *h_labels = new LabelT[graph.nodes];
  VertexT *h_preds = (mark_pred) ? new VertexT[graph.nodes] : NULL;

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  // perform BFS
  VertexT src;
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    src = srcs[run_num % num_srcs];
    GUARD_CU(problem.Reset(src, target));
    GUARD_CU(enactor.Reset(src, target));
    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact(src));
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, src = " + std::to_string(src) + ", #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);
    if (validation == "each") {
      GUARD_CU(problem.Extract(h_labels, h_preds));
      SizeT num_errors = app::bfs::Validate_Results(
          parameters, graph, src, h_labels, h_preds,
          ref_labels == NULL ? NULL : ref_labels[run_num % num_srcs], NULL,
          false);
    }
  }

  cpu_timer.Start();
  // Copy out results
  GUARD_CU(problem.Extract(h_labels, h_preds));
  if (validation == "last") {
    SizeT num_errors = app::bfs::Validate_Results(
        parameters, graph, src, h_labels, h_preds,
        ref_labels == NULL ? NULL : ref_labels[(num_runs - 1) % num_srcs]);
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, h_labels);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_labels;
  h_labels = NULL;
  delete[] h_preds;
  h_preds = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_bfs function
 * @tparam     GraphT     Type of the graph
 * @tparam     LabelT     Type of the labels
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] labels     Return shortest hop distance from source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename LabelT = typename GraphT::VertexT>
double gunrock_bfs(gunrock::util::Parameters &parameters, GraphT &graph,
                   LabelT **labels, typename GraphT::VertexT **preds = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::bfs::Problem<GraphT> ProblemT;
  typedef gunrock::app::bfs::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, target);
  enactor.Init(problem, target);

  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_runs = parameters.Get<int>("num-runs");
  int num_srcs = srcs.size();
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    int src_num = run_num % num_srcs;
    VertexT src = srcs[src_num];
    problem.Reset(src, target);
    enactor.Reset(src, target);

    cpu_timer.Start();
    enactor.Enact(src);
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(labels[src_num], preds == NULL ? NULL : preds[src_num]);
  }

  enactor.Release(target);
  problem.Release(target);
  srcs.clear();
  return total_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform SSSP
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[out] distances   Return shortest distance to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int,
          typename LabelT = VertexT>
double bfs(const SizeT num_nodes, const SizeT num_edges,
           const SizeT *row_offsets, const VertexT *col_indices,
           const int num_runs, VertexT *sources, const bool mark_pred,
           const bool direction_optimized, const bool idempotence,
           LabelT **labels, VertexT **preds = NULL) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, VertexT,
                                           gunrock::graph::HAS_CSR |
                                               gunrock::graph::HAS_CSC>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("bfs");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::bfs::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("mark-pred", mark_pred);
  parameters.Set("num-runs", num_runs);
  parameters.Set("direction-optimized", direction_optimized);
  parameters.Set("idempotence", idempotence);

  std::vector<VertexT> srcs;
  for (int i = 0; i < num_runs; i++) srcs.push_back(sources[i]);
  parameters.Set("srcs", srcs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer((SizeT *)row_offsets, num_nodes + 1,
                                     gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer((VertexT *)col_indices, num_edges,
                                        gunrock::util::HOST);
  graph.FromCsr(graph.csr(), gunrock::util::HOST, 0, quiet, true);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the BFS
  double elapsed_time = gunrock_bfs(parameters, graph, labels, preds);

  // Cleanup
  graph.Release();
  srcs.clear();

  return elapsed_time;
}

/*
 * @brief Simple C-interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[in]  direction_optimized Whether to use directional optimizing BFS
 * @param[in]  idempotence Whether to use idempotence
 * @param[out] labels      Return shortest hop distances to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
double bfs(const int num_nodes, const int num_edges, const int *row_offsets,
           const int *col_indices, int source, const bool mark_pred,
           const bool direction_optimized, const bool idempotence,
           int *distances, int *preds) {
  return bfs(num_nodes, num_edges, row_offsets, col_indices, 1, &source,
             mark_pred, direction_optimized, idempotence, &distances, &preds);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
