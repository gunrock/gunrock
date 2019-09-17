// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file pr_nibble_app.cu
 *
 * @brief Simple Gunrock Application
 */

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

#include <gunrock/app/pr_nibble/pr_nibble_enactor.cuh>
#include <gunrock/app/pr_nibble/pr_nibble_test.cuh>

namespace gunrock {
namespace app {
namespace pr_nibble {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  // app specific parameters
  GUARD_CU(parameters.Use<std::string>(
      "src",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "0",
      "<Vertex-ID|random|largestdegree> The source vertices\n"
      "\tIf random, randomly select non-zero degree vertices;\n"
      "\tIf largestdegree, select vertices with largest degrees",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "eps",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1e-2, "Convergence criteria.", __FILE__, __LINE__));
  GUARD_CU(parameters.Use<double>(
      "phi",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.5,
      "phi parameter",  // <TODO: DOCS>
      __FILE__, __LINE__));
  GUARD_CU(parameters.Use<double>(
      "vol",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      40.0,
      "volume parameter",  // <TODO: DOCS>
      __FILE__, __LINE__));
  GUARD_CU(parameters.Use<int>(
      "max-iter",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      3, "Max number of iterations", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run pr_nibble tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_values    Array of CPU reference values
 * @param[in]  target        where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::ValueT **ref_values,
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
  util::Info info("pr_nibble", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_srcs = srcs.size();

  ValueT *h_values = new ValueT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    auto run_index = run_num % num_srcs;
    VertexT src = srcs[run_index];
    VertexT src_neib = graph.GetEdgeDest(graph.GetNeighborListOffset(src));
    GUARD_CU(problem.Reset(src, src_neib, target));
    GUARD_CU(enactor.Reset(src, src_neib, target));

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
      GUARD_CU(problem.Extract(h_values));
      SizeT num_errors = Validate_Results(parameters, graph, h_values,
                                          ref_values[run_index], false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(h_values));
  if (validation == "last") {
    auto run_index = (num_runs - 1) % num_srcs;
    SizeT num_errors = Validate_Results(parameters, graph, h_values,
                                        ref_values[run_index], false);
  }

  // compute running statistics
  // <TODO> change NULL to problem specific per-vertex visited marker, e.g.
  // h_distances info.ComputeTraversalStats(enactor, (VertexT*)NULL);
  // Display_Memory_Usage(problem);
  // #ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
  // #endif
  // </TODO>

  // Clean up
  // GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_values;
  h_values = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace pr_nibble
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
double gunrock_pr_nibble(gunrock::util::Parameters &parameters, GraphT &graph,
                         ValueT *h_values) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::pr_nibble::Problem<GraphT> ProblemT;
  typedef gunrock::app::pr_nibble::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, target);
  enactor.Init(problem, target);

  int num_runs = parameters.Get<int>("num-runs");

  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_srcs = srcs.size();

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    auto run_index = run_num % num_srcs;

    VertexT src = srcs[run_index];
    VertexT src_neib = graph.GetEdgeDest(graph.GetNeighborListOffset(src));
    problem.Reset(src, src_neib, target);
    enactor.Reset(src, src_neib, target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();

    problem.Extract(h_values);
  }

  enactor.Release(target);
  problem.Release(target);
  srcs.clear();
  return total_time;
}

//  * @brief Simple interface take in graph as CSR format
//  * @param[in]  num_nodes   Number of veritces in the input graph
//  * @param[in]  num_edges   Number of edges in the input graph
//  * @param[in]  row_offsets CSR-formatted graph input row offsets
//  * @param[in]  col_indices CSR-formatted graph input column indices
//  * @param[in]  edge_values CSR-formatted graph input edge weights
//  * @param[in]  num_runs    Number of runs to perform SSSP
//  * @param[in]  sources     Sources to begin traverse, one for each run
//  * @param[in]  mark_preds  Whether to output predecessor info
//  * @param[out] distances   Return shortest distance to source per vertex
//  * @param[out] preds       Return predecessors of each vertex
//  * \return     double      Return accumulated elapsed times for all runs

template <typename VertexT = int, typename ValueT = float, typename SizeT = int,
          typename GValueT = unsigned int, typename TValueT = GValueT>
float pr_nibble(const SizeT num_nodes, const SizeT num_edges,
                const SizeT *row_offsets, const VertexT *col_indices,
                const GValueT *edge_values, const int num_runs,
                VertexT *sources, ValueT *h_values) {
  // TODO: change to other graph representation, if not using CSR
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("pr_nibble");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::pr_nibble::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("num-runs", num_runs);

  std::vector<VertexT> srcs;
  for (int i = 0; i < num_runs; i++) srcs.push_back(sources[i]);
  parameters.Set("srcs", srcs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;

  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, num_nodes + 1,
                                     gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, num_edges,
                                        gunrock::util::HOST);
  // graph.CsrT::edge_values   .SetPointer(edge_values, gunrock::util::HOST);
  // graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the pr_nibble
  double elapsed_time = gunrock_pr_nibble(parameters, graph, h_values);

  // Cleanup
  graph.Release();
  srcs.clear();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
