// //
// ----------------------------------------------------------------------------
// // Gunrock -- Fast and Efficient GPU Graph Library
// //
// ----------------------------------------------------------------------------
// // This source code is distributed under the terms of LICENSE.TXT
// // in the root directory of this source distribution.
// //
// ----------------------------------------------------------------------------

// /**
//  * @file bc_app.cu
//  *
//  * @brief Betweenness Centrality (BC) application
//  */

#include <iostream>
#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// betweenness centrality path includesls
#include <gunrock/app/bc/bc_enactor.cuh>
#include <gunrock/app/bc/bc_test.cuh>

namespace gunrock {
namespace app {
namespace bc {

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

  return retval;
}

/**
 * @brief Run BC tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_distances Reference distances
 * @param[in]  target        Where to perform the BC computation
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT,
          typename VertexT = typename GraphT::VertexT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,

                     ValueT **reference_bc_values = NULL,
                     ValueT **reference_sigmas = NULL,
                     VertexT **reference_labels = NULL,

                     util::Location target = util::DEVICE) {
  std::cout << "--- RunTests ---" << std::endl;

  cudaError_t retval = cudaSuccess;
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
  util::Info info("bc", parameters, graph);  // initialize Info structure

  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_srcs = srcs.size();

  // Allocate host-side array (for both reference and GPU-computed results)
  ValueT *h_bc_values = new ValueT[graph.nodes];
  ValueT *h_sigmas = new ValueT[graph.nodes];
  VertexT *h_labels = new VertexT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  // perform the algorithm
  VertexT src;
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    auto run_index = run_num % num_srcs;
    src = srcs[run_index];
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
      GUARD_CU(problem.Extract(h_bc_values, h_sigmas, h_labels));
      SizeT num_errors = app::bc::Validate_Results(
          parameters, graph, src, h_bc_values, h_sigmas, h_labels,
          reference_bc_values == NULL ? NULL : reference_bc_values[run_index],
          reference_sigmas == NULL ? NULL : reference_sigmas[run_index],
          reference_labels == NULL ? NULL : reference_labels[run_index], true);
    }
  }

  cpu_timer.Start();
  // Copy out results
  GUARD_CU(problem.Extract(h_bc_values, h_sigmas, h_labels));
  if (validation == "last") {
    auto run_index = (num_runs - 1) % num_srcs;
    SizeT num_errors = app::bc::Validate_Results(
        parameters, graph, src, h_bc_values, h_sigmas, h_labels,
        reference_bc_values == NULL ? NULL : reference_bc_values[run_index],
        reference_sigmas == NULL ? NULL : reference_sigmas[run_index],
        reference_labels == NULL ? NULL : reference_labels[run_index], true);
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, h_labels);
  // Display_Memory_Usage(problem);
  // #ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
  // #endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));

  delete[] h_bc_values;
  h_bc_values = NULL;
  delete[] h_sigmas;
  h_sigmas = NULL;
  delete[] h_labels;
  h_labels = NULL;

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace bc
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_bc function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the BC values
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] bc_values  Return betweenness centrality values per vertex
 * @param[out] sigmas     Return sigma of each vertex
 * @param[out] labels     Return label of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_bc(gunrock::util::Parameters &parameters, GraphT &graph,
                  ValueT **bc_values, ValueT **sigmas,
                  typename GraphT::VertexT **labels) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::bc::Problem<GraphT> ProblemT;
  typedef gunrock::app::bc::Enactor<ProblemT> EnactorT;
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
  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
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
    problem.Extract(bc_values[src_num], sigmas[src_num], labels[src_num]);
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
 * @param[in]  num_runs    Number of runs to perform BC
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[out] bc_values   Return betweenness centrality values per vertex
 * @param[out] sigmas      Return sigma of each vertex
 * @param[out] labels      Return label of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int,
          typename GValueT = float, typename BCValueT = GValueT>
float bc(const SizeT num_nodes, const SizeT num_edges, const SizeT *row_offsets,
         const VertexT *col_indices, const int num_runs, VertexT *sources,
         BCValueT **bc_values, BCValueT **sigmas, VertexT **labels) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("bc");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::bc::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("num-runs", num_runs);
  std::vector<VertexT> srcs;
  for (int i = 0; i < num_runs; i++) srcs.push_back(sources[i]);
  parameters.Set("srcs", srcs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;

  // Assign pointers into gunrock graph format
  CsrT csr;
  csr.Allocate(num_nodes, num_edges, gunrock::util::HOST);
  csr.row_offsets.SetPointer((SizeT *)row_offsets, num_nodes + 1,
                             gunrock::util::HOST);
  csr.column_indices.SetPointer((VertexT *)col_indices, num_edges,
                                gunrock::util::HOST);

  gunrock::graphio::LoadGraph(parameters, graph);

  // Run BC
  double elapsed_time =
      gunrock_bc(parameters, graph, bc_values, sigmas, labels);

  // Cleanup
  graph.Release();
  srcs.clear();

  return elapsed_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  num_runs    Number of runs to perform BC
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[out] bc_values   Return betweenness centrality values per vertex
 * @param[out] sigmas      Return sigma of each vertex
 * @param[out] labels      Return label of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
double bc(const int num_nodes, const int num_edges, const int *row_offsets,
          const int *col_indices, int source, float *bc_values, float *sigmas,
          int *labels) {
  return bc(num_nodes, num_edges, row_offsets, col_indices, 1 /* num_runs */,
            &source, &bc_values, &sigmas, &labels);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
