// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file mf_app.cu
 *
 * @brief maxflow (mf) application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/type_limits.cuh>

// Graph definations
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>
#include <gunrock/graphio/graphio.cuh>

// MF includes
#include <gunrock/app/mf/mf_enactor.cuh>
#include <gunrock/app/mf/mf_test.cuh>

//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a);\
    printf("\n");}
#define debug_aml(a...)

namespace gunrock {
namespace app {
namespace mf {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<uint64_t>(
      "source", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE,
      util::PreDefinedValues<uint64_t>::InvalidValue,
      "<Vertex-ID|random|largestdegree> The source vertex\n"
      "\tIf random, randomly select non-zero degree vertex;\n"
      "\tIf largestdegree, select vertex with largest degree",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint64_t>(
      "sink", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE,
      util::PreDefinedValues<uint64_t>::InvalidValue,
      "<Vertex-ID|random|largestdegree> The source vertex\n"
      "\tIf random, randomly select non-zero degree vertex;\n"
      "\tIf largestdegree, select vertex with largest degree",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-repeats",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "Number of repeats for ReapetFor operator\n"
      "\tDefault num-repeats is linear from number of vertices",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "seed to generate random sources or sink", __FILE__, __LINE__));
  return retval;
}

/**
 * @brief Run mf tests
 * @tparam     GraphT	  Type of the graph
 * @tparam     ValueT	  Type of the capacity on edges
 * @tparam     VertexT	  Type of vertex
 * @param[in]  parameters Excution parameters
 * @param[in]  graph	  Input graph
 * @param[in]  ref_flow	  Reference flow on edges
 * @param[in]  target	  Whether to perform the mf
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT, typename VertexT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     VertexT *h_reverse, ValueT *ref_flow, ValueT ref_max_flow,
                     util::Location target = util::DEVICE) {
  debug_aml("RunTests starts");
  cudaError_t retval = cudaSuccess;

  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  util::CpuTimer total_timer;
  total_timer.Start();
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  // parse configurations from parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  VertexT source = parameters.Get<VertexT>("source");
  VertexT sink = parameters.Get<VertexT>("sink");
  int num_repeats = parameters.Get<int>("num-repeats");
  debug_aml("source %d, sink %d, quite_mode %d, num-runs %d", source, sink,
            quiet_mode, num_runs);

  util::Info info("MF", parameters, graph);  // initialize Info structure

  // Allocate host-side array (for both reference and GPU-computed results)
  // ... for function Extract

  ValueT *h_flow = new ValueT[graph.edges];
  int *min_cut = new int[graph.nodes];
  // for (auto u = 0; u < graph.nodes; ++u) min_cut[u] = 0;
  memset(min_cut, 0, graph.nodes * sizeof(min_cut[0]));

  bool *vertex_reachabilities = new bool[graph.nodes];

  ValueT *h_residuals = new ValueT[graph.edges];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  // perform the MF algorithm
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(graph, h_reverse, target));
    GUARD_CU(enactor.Reset(source, target));

    util::PrintMsg("______GPU PushRelabel algorithm____", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    //    fprintf(stderr, "-----------------------------------\nRun %d, elapsed: %lf ms, #iterations = %d\n", \
		    run_num, cpu_timer.ElapsedMillis(), enactor.enactor_slices[0].enactor_stats.iteration);

    fprintf(stderr, "GPU Elapsed: %lf ms, ", cpu_timer.ElapsedMillis());
    util::PrintMsg(
        "-----------------------------------\nRun " + std::to_string(run_num) +
            ", elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);
    if (validation == "each") {
      GUARD_CU(problem.Extract(h_flow));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      app::mf::minCut(graph, source, h_flow, min_cut, vertex_reachabilities,
                      h_residuals);
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      int num_errors = app::mf::Validate_Results(
          parameters, graph, source, sink, h_flow, h_reverse, min_cut,
          ref_max_flow, ref_flow, quiet_mode);
    }
  }

  // Copy out results
  cpu_timer.Start();
  if (validation == "last") {
    GUARD_CU(problem.Extract(h_flow));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

    app::mf::minCut(graph, source, h_flow, min_cut, vertex_reachabilities,
                    h_residuals);
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

    int num_errors = app::mf::Validate_Results(
        parameters, graph, source, sink, h_flow, h_reverse, min_cut,
        ref_max_flow, ref_flow, quiet_mode);
  }

// Compute running statistics
// info.ComputeTraversalStats(enactor, h_flow);

// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));

  delete[] h_flow;
  h_flow = NULL;
  delete[] min_cut;
  min_cut = NULL;

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());

  return retval;
}

}  // namespace mf
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_maxflow function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the capacity/flow/excess
 *
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] flow	  Return flow on edges
 * @param[out] maxflow	  Return flow value
 * @param[out] min_cut	  Return partition into two sets of nodes
 * \return     double     Return accumulated elapsed times for all runs
 */
#if 0
template <typename GraphT, typename VertexT = typename GraphT::VertexT,
    typename ValueT = typename GraphT::ValueT>

double gunrock_mf(
    gunrock::util::Parameters &parameters,
    GraphT  &graph,
    VertexT *reverse,
    ValueT  *flow,
    int	    *min_cut,
    ValueT  &maxflow,
    bool   *vertex_reachabilities,
    ValueT *h_residuals)
{
    typedef gunrock::app::mf::Problem<GraphT>	ProblemT;
    typedef gunrock::app::mf::Enactor<ProblemT> EnactorT;

    gunrock::util::CpuTimer cpu_timer;
    gunrock::util::Location target = gunrock::util::DEVICE;

    double total_time = 0;
    if (parameters.UseDefault("quiet"))
        parameters.Set("quiet", true);

    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem(parameters);
    EnactorT enactor;
    problem.Init(graph,	  target);
    enactor.Init(problem, target);

    int num_runs = parameters.Get<int>("num-runs");
    int source = parameters.Get<VertexT>("source");
    int sink = parameters.Get<VertexT>("sink");

    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        problem.Reset(graph, reverse, target);
        enactor.Reset(source, target);

        cpu_timer.Start();
        enactor.Enact();
        cpu_timer.Stop();

        total_time += cpu_timer.ElapsedMillis();
        problem.Extract(flow);
	    gunrock::app::mf::minCut(graph, source, flow, min_cut, vertex_reachabilities, h_residuals);
    }

    enactor.Release(target);
    problem.Release(target);
    return total_time;
}
#endif

/*
 * @brief Simple interface  take in graph as CSR format
 * @param[in]  num_nodes    Number of veritces in the input graph
 * @param[in]  num_edges    Number of edges in the input graph
 * @param[in]  row_offsets  CSR-formatted graph input row offsets
 * @param[in]  col_indices  CSR-formatted graph input column indices
 * @param[in]  capacity	    CSR-formatted graph input edge weights
 * @param[in]  num_runs     Number of runs to perform mf
 * @param[in]  source	    Source to push flow towards the sink
 * @param[out] flow	    Return flow calculated on edges
 * @param[out] maxflow	    Return maxflow value
 * \return     double       Return accumulated elapsed times for all runs
 */
/*
template <
    typename VertexT  = uint32_t,
    typename SizeT    = uint32_t,
    typename ValueT   = double>
double mf(
        const int     num_runs,
        ValueT	      *flow,
        ValueT	      &maxflow,
        int	      *min_cut,
        int	      undirected = 0
        )
{
    typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
        gunrock::graph::HAS_EDGE_VALUES | gunrock::graph::HAS_CSR>  GraphT;
    typedef typename GraphT::CsrT				    CsrT;

    // Setup parameters
    gunrock::util::Parameters parameters("mf");
    gunrock::graphio::UseParameters(parameters);
    gunrock::app::mf::UseParameters(parameters);
    gunrock::app::UseParameters_test(parameters);
    parameters.Parse_CommandLine(0, NULL);
    parameters.Set("num-runs", num_runs);

    bool quiet = parameters.Get<bool>("quiet");

    GraphT d_graph;
    if (not undirected){
        parameters.Set<int>("remove-duplicate-edges", false);
        debug_aml("Load directed graph");
        gunrock::graphio::LoadGraph(parameters, d_graph);
    }

    GraphT u_graph;
    parameters.Set<int>("undirected", 1);
    parameters.Set<int>("remove-duplicate-edges", true);
    debug_aml("Load undirected graph");
    gunrock::graphio::LoadGraph(parameters, u_graph);

    if (parameters.Get<VertexT>("source") ==
            gunrock::util::PreDefinedValues<VertexT>::InvalidValue){
        parameters.Set("source", 0);
    }
    if (parameters.Get<VertexT>("sink") ==
            gunrock::util::PreDefinedValues<VertexT>::InvalidValue){
        parameters.Set("sink", u_graph.nodes-1);
    }

    VertexT* reverse = (VertexT*)malloc(sizeof(VertexT) * u_graph.edges);

    // Initialize reverse array.
    for (auto u = 0; u < u_graph.nodes; ++u)
    {
        auto e_start = u_graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = u_graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e)
        {
            auto v = u_graph.CsrT::GetEdgeDest(e);
            auto f_start = u_graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors2 = u_graph.CsrT::GetNeighborListLength(v);
            auto f_end = f_start + num_neighbors2;
            for (auto f = f_start; f < f_end; ++f)
            {
                auto z = u_graph.CsrT::GetEdgeDest(f);
                if (z == u)
                {
                    reverse[e] = f;
                    reverse[f] = e;
                    break;
                }
            }
        }
    }

    if (not undirected){
        // Correct capacity values on reverse edges
        for (auto u = 0; u < u_graph.nodes; ++u)
        {
            auto e_start = u_graph.CsrT::GetNeighborListOffset(u);
            auto num_neighbors = u_graph.CsrT::GetNeighborListLength(u);
            auto e_end = e_start + num_neighbors;
            for (auto e = e_start; e < e_end; ++e)
            {
                u_graph.CsrT::edge_values[e] = (ValueT)0;
                auto v = u_graph.CsrT::GetEdgeDest(e);
                // Looking for edge u->v in directed graph
                auto f_start = d_graph.CsrT::GetNeighborListOffset(u);
                auto num_neighbors2 = d_graph.CsrT::GetNeighborListLength(u);
                auto f_end = f_start + num_neighbors2;
                for (auto f = f_start; f < f_end; ++f)
                {
                    auto z = d_graph.CsrT::GetEdgeDest(f);
                    if (z == v and d_graph.CsrT::edge_values[f] > 0)
                    {
                        u_graph.CsrT::edge_values[e]  =
                            d_graph.CsrT::edge_values[f];
                        break;
                    }
                }
            }
        }
    }

    gunrock::util::Location target = gunrock::util::HOST;

    // Run the MF
    double elapsed_time = gunrock_mf(parameters, u_graph, reverse, flow,
            min_cut, maxflow);

    // Cleanup
    u_graph.Release();
    d_graph.Release();

    return elapsed_time;
}*/

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
