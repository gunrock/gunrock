// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sage_app.cuh
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
#include <gunrock/app/sage/sage_problem.cuh>
#include <gunrock/app/sage/sage_enactor.cuh>
#include <gunrock/app/sage/sage_test.cuh>

namespace gunrock {
namespace app {
namespace sage {

cudaError_t UseParameters(util::Parameters &parameters);

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
  // Display_Performance_Profiling(&enactor);
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
double gunrock_sage(gunrock::util::Parameters &parameters, GraphT &graph,
                    const ValueT *source_result,
                    gunrock::util::Location target = gunrock::util::DEVICE
                    // ValueT **distances,
                    // typename GraphT::VertexT **preds = NULL
) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::sage::Problem<GraphT> ProblemT;
  typedef gunrock::app::sage::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
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
    problem.Extract(source_result /*distances[src_num],
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
template <typename VertexT, typename SizeT, typename GValueT,
          typename SAGEValueT>
double sage(const SizeT num_nodes, const SizeT num_edges,
            const SizeT *row_offsets, const VertexT *col_indices,
            const GValueT *edge_values, const GValueT *source_result,
            const int num_runs,
            gunrock::util::Location allocated_on = gunrock::util::HOST
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
  gunrock::util::Location target = gunrock::util::HOST;

  if (allocated_on == gunrock::util::DEVICE) {
    target = gunrock::util::DEVICE;
  }

  graph.CsrT::Allocate(num_nodes, num_edges, target);
  graph.CsrT::row_offsets.SetPointer(row_offsets, num_nodes + 1, target);
  graph.CsrT::column_indices.SetPointer(col_indices, num_edges, target);
  graph.CsrT::edge_values.SetPointer(edge_values, num_edges, target);
  graph.FromCsr(graph.csr(), target, 0, quiet, true);
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the SSSP
  double elapsed_time = gunrock_sage(parameters, graph, source_result, target/*, distances, preds*/);

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
