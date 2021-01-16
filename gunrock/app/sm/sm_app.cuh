// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file ss_app.cuh
 *
 * @brief subgraph matching (SM) application header
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// subgraph matching includes
#include <gunrock/app/sm/sm_enactor.cuh>
#include <gunrock/app/sm/sm_test.cuh>

namespace gunrock {
namespace app {
namespace sm {

// template <typename ParametersT>
cudaError_t UseParameters(util::Parameters &parameters);

/**
 * @brief Run SM tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_distances Reference distances
 * @param[in]  target        Whether to perform the SM
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename VertexT = typename GraphT::VertexT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &data_graph,
                     GraphT &query_graph, unsigned long *ref_subgraphs,
                     util::Location target = util::DEVICE) {
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
  util::Info info("SM", parameters, data_graph);  // initialize Info structure

  unsigned long *count_subgraphs = new unsigned long[1];
  unsigned long **list_subgraphs = new unsigned long*[1];

  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(data_graph, query_graph, target));
  GUARD_CU(enactor.Init(problem, target));
  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  unsigned long num_subgraphs = 0;
  // perform SM
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(target));
    GUARD_CU(enactor.Reset(target));
    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);

    if (validation == "each") {
      GUARD_CU(problem.Extract(count_subgraphs, list_subgraphs));
      SizeT num_errors = app::sm::Validate_Results(
          parameters, data_graph, query_graph, count_subgraphs, ref_subgraphs,
          &num_subgraphs, false);
    }
  }

  cpu_timer.Start();
  // Copy out results
  GUARD_CU(problem.Extract(count_subgraphs, list_subgraphs));
  if (validation == "last") {
    SizeT num_errors = app::sm::Validate_Results(
        parameters, data_graph, query_graph, count_subgraphs, ref_subgraphs,
        &num_subgraphs, false);
  }

  UseParameters_test(parameters);
  parameters.Set("num-subgraphs", num_subgraphs);

  // compute running statistics
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(&enactor);
#endif
  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] count_subgraphs;
  count_subgraphs = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace sm
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_sm function
 * @tparam     GraphT          Type of the graph
 * @tparam     ValueT          Type of the distances
 * @param[in]  parameters      Excution parameters
 * @param[in]  data_graph      Input data graph
 * @param[in]  query_graph     Input query graph
 * @param[out] subgraphs       Return number of matched subgraphs
 * @param[out] list_subgraphs  Return list of matched subgraph combinations
 * @param[in]  allocated_on    Target device where inputs and outputs are stored
 * \return     double          Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_sm(gunrock::util::Parameters &parameters, GraphT &data_graph,
                  GraphT &query_graph, unsigned long *subgraphs,
                  unsigned long **list_subgraphs,
                  gunrock::util::Location allocated_on = gunrock::util::HOST) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::sm::Problem<GraphT> ProblemT;
  typedef gunrock::app::sm::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(data_graph, query_graph, target);
  enactor.Init(problem, target);

  int num_runs = parameters.Get<int>("num-runs");
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    problem.Reset(target);
    enactor.Reset(target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(subgraphs, list_subgraphs, target, allocated_on);
  }

  enactor.Release(target);
  problem.Release(target);
  return total_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes         Number of veritces in the input data graph
 * @param[in]  num_edges         Number of edges in the input data graph
 * @param[in]  row_offsets       CSR-formatted data graph input row offsets
 * @param[in]  col_indices       CSR-formatted data graph input column indices
 * @param[in]  num_query_nodes   Number of veritces in the input query graph
 * @param[in]  num_query_edges   Number of edges in the input query graph
 * @param[in]  query_row_offsets CSR-formatted query graph input row offsets
 * @param[in]  query_col_indices CSR-formatted query graph input column indices
 * @param[in]  num_runs          Number of runs to perform SM
 * @param[out] subgraphs         Return number of subgraphs
 * @param[out] list_subgraphs    Return list of subgraph combinations
 * @param[in]  allocated_on      Target device where inputs and outputs are stored
 * \return     double            Return accumulated elapsed times for all runs
 */
template <typename VertexT, typename SizeT>
double sm(const SizeT num_nodes, const SizeT num_edges,
                   const SizeT *row_offsets, const VertexT *col_indices,
                   const SizeT num_query_nodes, const SizeT num_query_edges,
                   const SizeT *query_row_offsets,
                   const VertexT *query_col_indices, const int num_runs,
                   unsigned long *subgraphs, unsigned long **list_subgraphs,
                   gunrock::util::Location allocated_on = gunrock::util::HOST) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, VertexT,
                                           gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("sm");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::sm::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("num-runs", num_runs);
  bool quiet = parameters.Get<bool>("quiet");
  GraphT data_graph;
  GraphT query_graph;

  gunrock::util::Location target = gunrock::util::HOST;

  if (allocated_on == gunrock::util::DEVICE) {
    target = gunrock::util::DEVICE;
  }

  // Assign pointers into gunrock graph format
  data_graph.CsrT::Allocate(num_nodes, num_edges, target);
  data_graph.CsrT::row_offsets.SetPointer((SizeT *)row_offsets, num_nodes + 1,
                                          target);
  data_graph.CsrT::column_indices.SetPointer((VertexT *)col_indices, num_edges,
                                             target);

  data_graph.FromCsr(data_graph.csr(), target, 0, quiet, true);

  query_graph.CsrT::Allocate(num_query_nodes, num_query_edges, target);
  query_graph.CsrT::row_offsets.SetPointer((SizeT *)query_row_offsets,
                                           num_query_nodes + 1, target);
  query_graph.CsrT::column_indices.SetPointer((VertexT *)query_col_indices,
                                              num_query_edges, target);

  query_graph.FromCsr(query_graph.csr(), target, 0, quiet, true);

  // Run the SM
  double elapsed_time =
      gunrock_sm(parameters, data_graph, query_graph, subgraphs, list_subgraphs, allocated_on);
  // Cleanup
  data_graph.Release();
  query_graph.Release();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
