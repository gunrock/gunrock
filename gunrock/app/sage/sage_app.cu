// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sage_app.cu
 *
 * @brief graphSage application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/sage/sage_test.cuh>
#include <gunrock/app/sage/sage_enactor.cuh>

namespace gunrock {
namespace app {
namespace sage {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "Wf1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^1 matrix in algorithm 2, feature part>\n"
      "\t dimension 64 by 128 for pokec;\n"
      "\t It should be child feature length by a value you want for W2 layer",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //     "Wf1-dim0",
  //     util::REQUIRED_ARGUMENT | util::SINGLE_VALUE |
  //     util::OPTIONAL_PARAMETER, 64, "Wf1 matrix row dimension",
  //     __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wf1-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wf1 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "Wa1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^1 matrix in algorithm 2, aggregation part>\n"
      "\t dimension 64 by 128 for pokec;\n"
      "\t It should be leaf feature length by a value you want for W2 layer",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //    "Wa1-dim0",
  //    util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
  //    64,
  //    "Wa1 matrix row dimension",
  //    __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wa1-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wa1 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "Wf2",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^2 matrix in algorithm 2, feature part>\n"
      "\t dimension 256 by 128 for pokec;\n"
      "\t It should be source_temp length by output length",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //     "Wf2-dim0",
  //     util::REQUIRED_ARGUMENT | util::SINGLE_VALUE |
  //     util::OPTIONAL_PARAMETER, 256, "Wf2 matrix row dimension",
  //     __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wf2-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wf2 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "Wa2",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<weight matrix for W^2 matrix in algorithm 2, aggregation part>\n"
      "\t dimension 256 by 128 for pokec;\n"
      "\t It should be child_temp length by output length",
      __FILE__, __LINE__));

  // GUARD_CU(parameters.Use<int>(
  //    "Wa2-dim0",
  //    util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
  //    256,
  //    "Wa2 matrix row dimension",
  //    __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "Wa2-dim1",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      128, "Wa2 matrix column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "features",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "",
      "<features matrix>\n"
      "\t dimension |V| by 64 for pokec;\n",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "feature-column",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      64, "feature column dimension", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-children-per-source",  // num_neigh1
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "number of sampled children per source", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-leafs-per-child",  // num_neight2
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "number of sampled leafs per child; default is the same as "
      "num-children-per-source",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "batch-size",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      65536, "number of source vertex to process in one iteration", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<int>(
      "rand-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "seed for random number generator; default will use time(NULL)", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "custom-kernels",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      true, "whether to use custom CUDA kernels", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "omp-threads",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      32, "number of threads to run CPU reference", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run Sage tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_distances Reference distances
 * @param[in]  target        Whether to perform the Sage
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     // ValueT **ref_distances = NULL,
                     util::Location target = util::DEVICE) {
  cudaError_t retval = cudaSuccess;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  // typedef typename GraphT::ValueT  ValueT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;
  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // parse configurations from parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("Sage", parameters, graph);  // initialize Info structure

  // Allocate host-side array (for both reference and GPU-computed results)

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  // util::PrintMsg("Before init");
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));
  ValueT *h_source_result = new ValueT[((uint64_t)graph.nodes) *
                                       problem.data_slices[0][0].result_column];
  // util::PrintMsg("After init");
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());
  // info.preprocess_time = cpu_timer.ElapsedMillis();

  // perform SAGE
  // VertexT src;
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    // src = srcs[run_num % num_srcs];
    GUARD_CU(problem.Reset(target));
    GUARD_CU(enactor.Reset(target));
    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " +
            std::to_string(cpu_timer.ElapsedMillis())
            //+ " ms, src = "+ std::to_string(src)
            + " ms, #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);
    if (validation == "each") {
      GUARD_CU(problem.Extract(h_source_result));
      SizeT num_errors = app::sage::Validate_Results(
          parameters, graph, h_source_result,
          problem.data_slices[0][0].result_column, false);
    }
  }

  cpu_timer.Start();
  // Copy out results
  GUARD_CU(problem.Extract(h_source_result));
  if (validation == "last") {
    SizeT num_errors = app::sage::Validate_Results(
        parameters, graph, h_source_result,
        problem.data_slices[0][0].result_column, true);
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
  delete[] h_source_result;
  h_source_result = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace sage
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_sage function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the distances
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] distances  Return shortest distance to source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_sage(gunrock::util::Parameters &parameters, GraphT &graph
                    // ValueT **distances,
                    // typename GraphT::VertexT **preds = NULL
) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::sage::Problem<GraphT> ProblemT;
  typedef gunrock::app::sage::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, target);
  enactor.Init(problem, target);

  // std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
  int num_runs = parameters.Get<int>("num-runs");
  // int num_srcs = srcs.size();
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    // int src_num = run_num % num_srcs;
    // VertexT src = srcs[src_num];
    problem.Reset(target);
    enactor.Reset(target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract( /*distances[src_num],
            preds == NULL ? NULL : preds[src_num]*/);
  }

  enactor.Release(target);
  problem.Release(target);
  // srcs.clear();
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
          typename GValueT = unsigned int, typename SAGEValueT = GValueT>
double sage(const SizeT num_nodes, const SizeT num_edges,
            const SizeT *row_offsets, const VertexT *col_indices,
            const GValueT *edge_values, const int num_runs
            //      VertexT     *sources,
            // const bool         mark_pred,
            //      SSSPValueT **distances,
            //      VertexT    **preds = NULL
) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("sage");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::sage::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  // parameters.Set("mark-pred", mark_pred);
  parameters.Set("num-runs", num_runs);
  // std::vector<VertexT> srcs;
  // for (int i = 0; i < num_runs; i ++)
  //    srcs.push_back(sources[i]);
  // parameters.Set("srcs", srcs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, num_nodes + 1,
                                     gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, num_edges,
                                        gunrock::util::HOST);
  graph.CsrT::edge_values.SetPointer(edge_values, num_edges,
                                     gunrock::util::HOST);
  // graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the SSSP
  double elapsed_time = gunrock_sage(parameters, graph /*, distances, preds*/);

  // Cleanup
  graph.Release();
  // srcs.clear();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
