// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file pr_app.cu
 *
 * @brief Gunrock PageRank application
 */

// <primitive>_app.cuh includes
#include <gunrock/app/app.cuh>

// page-rank includes
#include <gunrock/app/pr/pr_enactor.cuh>
#include <gunrock/app/pr/pr_test.cuh>

namespace gunrock {
namespace app {
namespace pr {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "src",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "invalid",
      "<Vertex-ID|random|largestdegree|invalid> The source vertices\n"
      "\tIf random, randomly select non-zero degree vertices;\n"
      "\tIf largestdegree, select vertices with largest degrees;\n"
      "\tIf invalid, do not use personalized PageRank.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "src-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "seed to generate random sources", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "output-filename",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", "file to output ranking values", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run PageRank tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_node_ids  Reference top-ranked vertex IDs
 * @param[in]  ref_ranks     Reference ranking values per vertex
 * @param[in]  target        Whether to perform the PageRank
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::VertexT **ref_node_ids = NULL,
                     ValueT **ref_ranks = NULL,
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
  bool quiet = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_srcs = srcs.size();
  util::Info info("PR", parameters, graph);  // initialize Info structure

  // Allocate host-side array (for both reference and GPU-computed results)
  ValueT *h_ranks = new ValueT[graph.nodes];
  VertexT *h_node_ids = new VertexT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  // util::PrintMsg("Before init");
  GUARD_CU(problem.Init(graph, target));
  // util::PrintMsg("Problem init");
  GUARD_CU(enactor.Init(problem, target));
  // util::PrintMsg("After init");
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  // perform PageRank
  VertexT src;
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    src = srcs[run_num % num_srcs];
    GUARD_CU(problem.Reset(src, target));
    GUARD_CU(enactor.Reset(src, target));
    util::PrintMsg("__________________________", !quiet);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact(src));
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, src = " + std::to_string(src) + ", #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet);
    if (validation == "each") {
      GUARD_CU(enactor.Extract());
      GUARD_CU(problem.Extract(h_node_ids, h_ranks));
      ValueT total_rank = 0;
#pragma omp parallel for reduction(+ : total_rank)
      for (VertexT v = 0; v < graph.nodes; v++) {
        total_rank += h_ranks[v];
      }
      util::PrintMsg("Total_rank = " + std::to_string(total_rank));

      SizeT num_errors = app::pr::Validate_Results(
          parameters, graph, src, h_node_ids, h_ranks,
          ref_node_ids == NULL ? NULL : ref_node_ids[run_num % num_srcs],
          ref_ranks == NULL ? NULL : ref_ranks[run_num % num_srcs], false);
    }
  }

  cpu_timer.Start();
  if (validation == "last") {
    // Copy out results
    GUARD_CU(enactor.Extract());
    GUARD_CU(problem.Extract(h_node_ids, h_ranks));
    if (!quiet) {
      ValueT total_rank = 0;
#pragma omp parallel for reduction(+ : total_rank)
      for (VertexT v = 0; v < graph.nodes; v++) {
        total_rank += h_ranks[v];
      }
      util::PrintMsg("Total_rank = " + std::to_string(total_rank));

      // Display Solution
      DisplaySolution(h_node_ids, h_ranks, graph.nodes);
    }
    SizeT num_errors = app::pr::Validate_Results(
        parameters, graph, src, h_node_ids, h_ranks,
        ref_node_ids == NULL ? NULL : ref_node_ids[(num_runs - 1) % num_srcs],
        ref_ranks == NULL ? NULL : ref_ranks[(num_runs - 1) % num_srcs], false);
  }

  if (parameters.Get<std::string>("output-filename") != "") {
    cpu_timer.Start();
    std::ofstream fout;
    size_t buf_size = 1024 * 1024 * 16;
    char *fout_buf = new char[buf_size];
    fout.rdbuf()->pubsetbuf(fout_buf, buf_size);
    fout.open(parameters.Get<std::string>("output-filename").c_str());

    for (VertexT v = 0; v < graph.nodes; v++) {
      fout << h_node_ids[v] + 1 << "," << h_ranks[v] << std::endl;
    }
    fout.close();
    delete[] fout_buf;
    fout_buf = NULL;
    cpu_timer.Stop();
    parameters.Set("write-time", cpu_timer.ElapsedMillis());
  }

// compute running statistics
// info.ComputeTraversalStats(enactor, h_distances);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_node_ids;
  h_node_ids = NULL;
  delete[] h_ranks;
  h_ranks = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace pr
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_sssp function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the ranking values
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] node_ids   Return top-ranked vertex IDs
 * @param[out] ranks      Return PageRank scores per node
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_pagerank(gunrock::util::Parameters &parameters, GraphT &graph,
                        typename GraphT::VertexT **node_ids, ValueT **ranks) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::pr::Problem<GraphT> ProblemT;
  typedef gunrock::app::pr::Enactor<ProblemT> EnactorT;

  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;

  printf("Init Problem and Enactor for PR.\n");
  problem.Init(graph, target);
  enactor.Init(problem, target);

  std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_runs = parameters.Get<int>("num-runs");
  int num_srcs = srcs.size();
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    printf("For run_num: %d, Reset problem and enactor and Enact.\n", run_num);
    int src_num = run_num % num_srcs;
    VertexT src = srcs[src_num];
    problem.Reset(src, target);
    enactor.Reset(src, target);

    cpu_timer.Start();
    enactor.Enact(src);
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    enactor.Extract();
    problem.Extract(node_ids[src_num], ranks[src_num]);
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
 * @param[in]  num_runs    Number of runs to perform PR
 * @param[in]  sources     Sources for personalized PR
 * @param[in]  normalize   Whether to normalize ranking values
 * @param[out] node_ids    Return top-ranked vertex IDs
 * @param[out] pagerank    Return PageRank scores per node
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int, typename ValueT = float>
double pagerank(const SizeT num_nodes, const SizeT num_edges,
                const SizeT *row_offsets, const VertexT *col_indices,
                const int num_runs, bool normalize, VertexT *sources,
                VertexT **node_ids, ValueT **ranks) {
  typedef typename gunrock::app::TestGraph<
      VertexT, SizeT, ValueT, gunrock::graph::HAS_COO | gunrock::graph::HAS_CSC>
      GraphT;
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
                                           gunrock::graph::HAS_CSR>
      Graph_CsrT;
  typedef typename Graph_CsrT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("pr");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::pr::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("normalize", normalize);
  parameters.Set("num-runs", num_runs);
  std::vector<VertexT> srcs;
  VertexT InvalidValue = gunrock::util::PreDefinedValues<VertexT>::InvalidValue;

  for (int i = 0; i < num_runs; i++) {
    if (sources != NULL)
      srcs.push_back(sources[i]);
    else
      srcs.push_back(InvalidValue);
  }
  parameters.Set("srcs", srcs);

  bool quiet = parameters.Get<bool>("quiet");

  CsrT csr;
  // Assign pointers into gunrock graph format
  csr.Allocate(num_nodes, num_edges, gunrock::util::HOST);
  csr.row_offsets.SetPointer((int *)row_offsets, num_nodes + 1,
                             gunrock::util::HOST);
  csr.column_indices.SetPointer((int *)col_indices, num_edges,
                                gunrock::util::HOST);
  // csr.Move(gunrock::util::HOST, gunrock::util::DEVICE);

  gunrock::util::Location target = gunrock::util::HOST;

  GraphT graph;
  graph.FromCsr(csr, target, 0, quiet, true);
  csr.Release();
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the PR
  double elapsed_time = gunrock_pagerank(parameters, graph, node_ids, ranks);

  // Cleanup
  // graph.Release();
  // srcs.clear();

  return elapsed_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source for personalized PR
 * @param[in]  normalize   Whether to normalize ranking values
 * @param[out] node_ids    Return top-ranked vertex IDs
 * @param[out] pagerank    Return PageRank scores per node
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int, typename ValueT = float>
double pagerank(const SizeT num_nodes, const SizeT num_edges,
                const SizeT *row_offsets, const VertexT *col_indices,
                bool normalize, VertexT source, VertexT *node_ids,
                ValueT *ranks) {
  if (source == -1) {
    return pagerank(num_nodes, num_edges, row_offsets, col_indices,
                    1 /* num_runs */, normalize, (int *)NULL, &node_ids,
                    &ranks);
  }

  return pagerank(num_nodes, num_edges, row_offsets, col_indices,
                  1 /* num_runs */, normalize, &source, &node_ids, &ranks);
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  normalize   Whether to normalize ranking values
 * @param[out] node_ids    Return top-ranked vertex IDs
 * @param[out] pagerank    Return PageRank scores per node
 * \return     double      Return accumulated elapsed times for all runs
 */
double pagerank(const int num_nodes, const int num_edges,
                const int *row_offsets, const int *col_indices, bool normalize,
                int *node_ids, float *ranks) {
  return pagerank(num_nodes, num_edges, row_offsets, col_indices, normalize,
                  (int)-1 /* source */, node_ids, ranks);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
