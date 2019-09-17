// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file vn_app.cu
 *
 * @brief single-source shortest path (vn) application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/vn/vn_enactor.cuh>
#include <gunrock/app/vn/vn_test.cuh>

namespace gunrock {
namespace app {
namespace vn {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "src",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "0",
      "<Vertex-IDs|random|largestdegree> The source vertices\n"
      "\tIf random, randomly select non-zero degree vertices;\n"
      "\tIf largestdegree, select vertices with largest degrees",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "src-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "seed to generate random sources", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "srcs-per-run",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "number of source nodes per run", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run vn tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_distances Reference distances
 * @param[in]  target        Whether to perform the vn
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT,
          typename VertexT = typename GraphT::VertexT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     ValueT **ref_distances = NULL,
                     util::Location target = util::DEVICE) {
  cudaError_t retval = cudaSuccess;
  // typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;
  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // parse configurations from parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  bool mark_pred = parameters.Get<bool>("mark-pred");
  std::string validation = parameters.Get<std::string>("validation");

  // Load srcs
  std::vector<VertexT> srcs_vector =
      parameters.Get<std::vector<VertexT>>("srcs");
  int total_num_srcs = srcs_vector.size();
  int num_runs = parameters.Get<int>("num-runs");
  int srcs_per_run = parameters.Get<int>("srcs-per-run");
  if (srcs_per_run == util::PreDefinedValues<int>::InvalidValue) {
    srcs_per_run = total_num_srcs;
  }
  assert(total_num_srcs == num_runs * srcs_per_run);
  VertexT *all_srcs = &srcs_vector[0];

  util::Info info("vn", parameters, graph);  // initialize Info structure

  // Allocate host-side array (for both reference and GPU-computed results)
  ValueT *h_distances = new ValueT[graph.nodes];
  VertexT *h_preds = (mark_pred) ? new VertexT[graph.nodes] : NULL;

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());
  // info.preprocess_time = cpu_timer.ElapsedMillis();

  VertexT *srcs = new VertexT[srcs_per_run];
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    for (SizeT i = 0; i < srcs_per_run; ++i) {
      srcs[i] = all_srcs[run_num * srcs_per_run + i % total_num_srcs];
    }

    GUARD_CU(problem.Reset(srcs, srcs_per_run, target));
    GUARD_CU(enactor.Reset(srcs, srcs_per_run, target));
    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact(srcs));
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    std::string src_msg = "";
    for (SizeT i = 0; i < srcs_per_run; ++i) {
      src_msg += std::to_string(srcs[i]);
      if (i != srcs_per_run - 1) src_msg += ",";
    }
    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, srcs = " + src_msg + ", #iterations = "  // TODO -- fix docs
            + std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);

    if (validation == "each") {
      GUARD_CU(problem.Extract(h_distances, h_preds));
      SizeT num_errors = app::vn::Validate_Results(
          parameters, graph, srcs, h_distances, h_preds,
          ref_distances == NULL ? NULL : ref_distances[run_num], NULL, false);
    }
  }

  cpu_timer.Start();
  GUARD_CU(problem.Extract(h_distances, h_preds));
  if (validation == "last") {
    SizeT num_errors = app::vn::Validate_Results(
        parameters, graph, srcs, h_distances, h_preds,
        ref_distances == NULL ? NULL : ref_distances[num_runs - 1]);
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, h_distances);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_distances;
  h_distances = NULL;
  delete[] h_preds;
  h_preds = NULL;
  delete[] all_srcs;
  all_srcs = NULL;
  delete[] srcs;
  srcs = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace vn
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_vn function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the distances
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] distances  Return shortest distance to source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_vn(gunrock::util::Parameters &parameters, GraphT &graph,
                  ValueT *distances, typename GraphT::VertexT *preds = NULL) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef gunrock::app::vn::Problem<GraphT> ProblemT;
  typedef gunrock::app::vn::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, target);
  enactor.Init(problem, target);

  std::vector<VertexT> srcs_vector =
      parameters.Get<std::vector<VertexT>>("srcs");
  SizeT num_srcs = srcs_vector.size();
  VertexT *srcs = new VertexT[num_srcs];
  for (SizeT i = 0; i < num_srcs; ++i) {
    srcs[i] = srcs_vector[i];
  }

  int num_runs = parameters.Get<int>("num-runs");
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    problem.Reset(srcs, num_srcs, target);
    enactor.Reset(srcs, num_srcs, target);

    cpu_timer.Start();
    enactor.Enact(srcs);
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(distances, preds == NULL ? NULL : preds);
  }

  enactor.Release(target);
  problem.Release(target);
  srcs_vector.clear();
  return total_time;
}

//  * @brief Simple interface take in graph as CSR format
//  * @param[in]  num_nodes   Number of veritces in the input graph
//  * @param[in]  num_edges   Number of edges in the input graph
//  * @param[in]  row_offsets CSR-formatted graph input row offsets
//  * @param[in]  col_indices CSR-formatted graph input column indices
//  * @param[in]  edge_values CSR-formatted graph input edge weights
//  * @param[in]  num_runs    Number of runs to perform vn
//  * @param[in]  sources     Sources to begin traverse, one for each run
//  * @param[in]  mark_preds  Whether to output predecessor info
//  * @param[out] distances   Return shortest distance to source per vertex
//  * @param[out] preds       Return predecessors of each vertex
//  * \return     double      Return accumulated elapsed times for all runs

template <typename VertexT = int, typename SizeT = int,
          typename GValueT = unsigned int, typename vnValueT = GValueT>
double vn(const SizeT num_nodes, const SizeT num_edges,
          const SizeT *row_offsets, const VertexT *col_indices,
          const GValueT *edge_values, const int num_runs, VertexT *sources,
          const bool mark_pred, vnValueT *distances, VertexT *preds = NULL) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("vn");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::vn::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("mark-pred", mark_pred);
  parameters.Set("num-runs", num_runs);
  std::vector<VertexT> srcs;
  for (int i = 0; i < num_runs; i++) srcs.push_back(sources[i]);

  parameters.Set("srcs", srcs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, num_nodes + 1,
                                     gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, num_edges,
                                        gunrock::util::HOST);
  // graph.CsrT::edge_values   .SetPointer(edge_values, gunrock::util::HOST);
  // graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the vn
  double elapsed_time = gunrock_vn(parameters, graph, distances, preds);

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
