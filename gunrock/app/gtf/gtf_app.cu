// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file gtf_app.cu
 *
 * @brief maxflow (gtf) application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/type_limits.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// gtf includes
#include <gunrock/app/gtf/gtf_enactor.cuh>
#include <gunrock/app/gtf/gtf_test.cuh>

//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a);\
    printf("\n");}
#define debug_aml(a...)

namespace gunrock {
namespace app {
namespace gtf {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  GUARD_CU(UseParameters_test(parameters));

  GUARD_CU(parameters.Use<uint64_t>(
      "source", util::INTERNAL_PARAMETER | util::REQUIRED_ARGUMENT,
      util::PreDefinedValues<uint64_t>::InvalidValue,
      "<Vertex-ID|random|largestdegree> The source vertex\n"
      "\tIf random, randomly select non-zero degree vertex;\n"
      "\tIf largestdegree, select vertex with largest degree",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint64_t>(
      "sink", util::INTERNAL_PARAMETER | util::REQUIRED_ARGUMENT,
      util::PreDefinedValues<uint64_t>::InvalidValue,
      "<Vertex-ID|random|largestdegree> The source vertex\n"
      "\tIf random, randomly select non-zero degree vertex;\n"
      "\tIf largestdegree, select vertex with largest degree",
      __FILE__, __LINE__));

  return retval;
}

template <typename GraphT, typename ArrayT, typename ValueT>
cudaError_t AddSourceSink(GraphT &u_graph, ArrayT &weights, GraphT &graph,
                          ValueT lambda) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(graph.Allocate(u_graph.nodes + 2, u_graph.edges + u_graph.nodes * 4,
                          util::HOST));
  printf("# for nodes and edges (%d -> %d)\n", u_graph.nodes + 2,
         u_graph.edges + u_graph.nodes * 4);

  typedef typename GraphT::CsrT CsrT;
  printf("in add source sink # of nodes %d \n", u_graph.nodes);

  for (auto i = 0; i < graph.edges; i++) {
    if (i < graph.nodes) graph.row_offsets[i] = 100;
    graph.column_indices[i] = 100;
  }
  for (auto u = 0; u < u_graph.nodes; ++u) {
    auto e_start = u_graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = u_graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      auto v = u_graph.CsrT::GetEdgeDest(e);
      printf("flow(%d -> %d) = %lf\n", u, v, graph.edge_values[e]);
    }
  }
  for (int u = 0; u < u_graph.nodes; ++u)
    printf("in add source sink weights %f \n", weights[u]);

  for (auto v = 0; v < u_graph.nodes; v++) {
    auto e_start = u_graph.GetNeighborListOffset(v);
    auto num_neighbors = u_graph.GetNeighborListLength(v);
    auto e_end = e_start + num_neighbors;

    graph.row_offsets[v] = u_graph.row_offsets[v] + v * 2;
    // printf("!!!!%d row_offsets %d\n", v, u_graph.row_offsets[v] + v * 2);
    for (auto e = e_start; e < e_end; e++) {
      graph.edge_values[e + v * 2] = u_graph.edge_values[e];
    }
    graph.column_indices[e_end + v * 2] = u_graph.nodes;
    graph.edge_values[e_end + v * 2] = 0;
    graph.column_indices[e_end + v * 2 + 1] = u_graph.nodes + 1;
    graph.edge_values[e_end + v * 2 + 1] = 0;
    printf("!!!!%d  %d, edge_values\n", v, e_end + v * 2);
  }
  for (auto v = u_graph.nodes; v < u_graph.nodes + 3; v++)
    graph.row_offsets[u_graph.nodes + v] =
        u_graph.edges + u_graph.nodes * (2 + v);

  for (auto u = 0; u < graph.nodes; ++u) {
    auto e_start = graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      auto v = graph.CsrT::GetEdgeDest(e);
      printf("intermediate graph (%d -> %d) = %lf\n", u, v,
             graph.edge_values[e]);
    }
  }

  auto offset = u_graph.edges + u_graph.nodes * 2;
  for (auto v = 0; v < u_graph.nodes; v++) {
    graph.column_indices[offset + v] = v;
    graph.edge_values[offset + v] = 0;
    graph.column_indices[offset + v + u_graph.nodes] = v;
    graph.edge_values[offset + v + u_graph.nodes] = 0;
  }

  for (auto v = 0; v < u_graph.nodes; v++) {
    auto weight = weights[v];
    if (weight < 0) {  // weight to the sink
      graph.edge_values[graph.row_offsets[v] + u_graph.nodes + 1] = -1 * weight;
    } else {  // weight from the source
      graph.edge_values[u_graph.edges + u_graph.nodes * 2 + v] = weight;
    }
  }
  for (auto u = 0; u < graph.nodes; ++u) {
    auto e_start = graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      auto v = graph.CsrT::GetEdgeDest(e);
      printf("new graph (%d -> %d) = %lf\n", u, v, graph.edge_values[e]);
    }
  }
  return retval;
}

/**
 * @brief Run gtf tests
 * @tparam     GraphT	  Type of the graph
 * @tparam     ValueT	  Type of the capacity on edges
 * @tparam     VertexT	  Type of vertex
 * @param[in]  parameters Excution parameters
 * @param[in]  graph	  Input graph
 * @param[in]  ref_flow	  Reference flow on edges
 * @param[in]  target	  Whether to perform the gtf
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT, typename VertexT, typename SizeT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     SizeT *h_reverse, util::Location target = util::DEVICE) {
  debug_aml("RunTests starts");
  cudaError_t retval = cudaSuccess;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  util::CpuTimer total_timer;
  total_timer.Start();
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  // parse configurations from parameters
  printf("in side that weird function1\n");
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");

  ///////////////////////////////
  auto num_nodes = graph.nodes;        // n + 2 = V
  auto num_org_nodes = num_nodes - 2;  // n
  auto num_edges = graph.edges;        // m + n*4
  VertexT source = num_org_nodes;      // originally 0

  SizeT offset = num_edges - num_org_nodes * 2;
  printf("offset in GPU preprocess is %d num edges %d \n", offset, num_edges);
  double sum_weights_source_sink = 0;
  for (VertexT v = 0; v < num_org_nodes; v++) {
    sum_weights_source_sink += graph.edge_values[offset + v];
    SizeT e =
        graph.GetNeighborListOffset(v) + graph.GetNeighborListLength(v) - 1;
    sum_weights_source_sink -= graph.edge_values[e];
  }

  auto avg_weights_source_sink = sum_weights_source_sink / num_org_nodes;
  printf("avg in GPU preprocess is %f \n", avg_weights_source_sink);
  for (VertexT v = 0; v < num_org_nodes; v++) {
    SizeT e =
        graph.GetNeighborListOffset(v) + graph.GetNeighborListLength(v) - 1;
    ValueT val = graph.edge_values[offset + v] - graph.edge_values[e] -
                 avg_weights_source_sink;

    if (val > 0) {
      graph.edge_values[offset + v] = val;
      graph.edge_values[e] = 0;
    } else {
      graph.edge_values[offset + v] = 0;
      graph.edge_values[e] = -1 * val;
    }
  }
  ///////////////////////////////

  debug_aml("source %d, sink %d, quite_mode %d, num-runs %d", source, sink,
            quiet_mode, num_runs);
  util::Info info("gtf", parameters, graph);  // initialize Info structure

  // Allocate host-side array (for both reference and GPU-computed results)
  // ... for function Extract
  util::Array1D<SizeT, ValueT> community_accus;
  community_accus.SetName("community_accus");
  GUARD_CU(community_accus.Allocate(graph.nodes, util::HOST | util::DEVICE));

  community_accus[0] = avg_weights_source_sink;

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  // perform the gtf algorithm
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(graph, community_accus + 0, h_reverse, target));
    GUARD_CU(enactor.Reset(source, target));

    util::PrintMsg("______GPU SFL algorithm____", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "-----------------------------------\nRun " + std::to_string(run_num) +
            ", elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);

    ValueT *Y = (ValueT *)malloc(sizeof(ValueT) * graph.nodes);
    ValueT *edge_values = (ValueT *)malloc(sizeof(ValueT) * graph.edges);
    GUARD_CU(problem.Extract(Y, edge_values));

    std::ofstream out_pr("./output_pr_GPU.txt");
    for (int i = 0; i < num_org_nodes; i++) out_pr << Y[i] << std::endl;
    out_pr.close();

    std::ofstream out_graph("./wrong_graph_GPU.mtx");
    out_graph << graph.nodes << ' ' << graph.nodes << ' ' << graph.edges
              << std::endl;
    for (VertexT v = 0; v < graph.nodes; v++) {
      auto e_start = graph.GraphT::CsrT::GetNeighborListOffset(v);
      auto num_neighbors = graph.GraphT::CsrT::GetNeighborListLength(v);
      auto e_end = e_start + num_neighbors;
      for (auto e = e_start; e < e_end; ++e) {
        VertexT u = graph.GetEdgeDest(e);
        out_graph << v + 1 << ' ' << u + 1 << ' ' << graph.edge_values[e]
                  << std::endl;
      }
    }

    out_graph.close();

    // if (validation == "each")
    // {

    // int num_errors = app::gtf::Validate_Results(parameters, graph,
    // source, sink, h_flow, h_reverse, ref_flow, quiet_mode);
    // }
  }

  // Copy out results
  cpu_timer.Start();
  if (validation == "last") {
    // GUARD_CU(problem.Extract(h_flow)); //!!!
    /*for (int i=0; i<graph.edges; ++i){
        if (ref_flow){
        debug_aml("h_flow[%d]=%lf, ref_flow[%d] = %lf",
              i, h_flow[i], i, ref_flow[i]);
        }
    }*/
    // int num_errors = app::gtf::Validate_Results(parameters, graph,
    // source, sink, h_flow, h_reverse, ref_flow, quiet_mode); //!!!
  }

// Compute running statistics
// info.ComputeTraversalStats(enactor, h_flow);//!!!
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  // delete[] h_flow;
  // h_flow = NULL; //!!!

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());

  return retval;
}

}  // namespace gtf
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_maxflow function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the capacity/flow/excess
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] flow	  Return
 * @param[out] maxflow	  Return
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_gtf(gunrock::util::Parameters &parameters, GraphT &graph,
                   ValueT *flow, ValueT &maxflow) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::gtf::Problem<GraphT> ProblemT;
  typedef gunrock::app::gtf::Enactor<ProblemT> EnactorT;
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
  int source = parameters.Get<VertexT>("source");
  int sink = parameters.Get<VertexT>("sink");

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    problem.Reset(target);
    enactor.Reset(target);

    cpu_timer.Start();
    enactor.Enact(source, sink);
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(flow);
  }

  enactor.Release(target);
  problem.Release(target);
  return total_time;
}

/*
 * @brief Simple interface  take in graph as CSR format
 * @param[in]  num_nodes    Number of veritces in the input graph
 * @param[in]  num_edges    Number of edges in the input graph
 * @param[in]  row_offsets  CSR-formatted graph input row offsets
 * @param[in]  col_indices  CSR-formatted graph input column indices
 * @param[in]  capacity	    CSR-formatted graph input edge weights
 * @param[in]  num_runs     Number of runs to perform gtf
 * @param[in]  source	    Source to push flow towards the sink
 * @param[out] flow	    Return flow calculated on edges
 * @param[out] maxflow	    Return maxflow value
 * \return     double       Return accumulated elapsed times for all runs
 */
template <typename VertexT = uint32_t, typename SizeT = uint32_t,
          typename ValueT = double>
float gtf(const SizeT num_nodes, const SizeT num_edges,
          const SizeT *row_offsets, const VertexT *col_indices,
          const ValueT capacity, const int num_runs, VertexT source,
          VertexT sink, ValueT *flow, ValueT &maxflow) {
  // TODO: change to other graph representation, if not using CSR
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_COO>
      GraphT;
  typedef typename GraphT::CooT CooT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("gtf");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::gtf::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("num-runs", num_runs);
  parameters.Set("source", source);
  parameters.Set("sink", sink);

  bool quiet = parameters.Get<bool>("quiet");
  CsrT csr;
  // Assign pointers into gunrock graph format
  csr.Allocate(num_nodes, num_edges, gunrock::util::HOST);
  csr.row_offsets.SetPointer(row_offsets, gunrock::util::HOST);
  csr.column_indices.SetPointer(col_indices, gunrock::util::HOST);
  csr.capacity.SetPointer(capacity, gunrock::util::HOST);

  gunrock::util::Location target = gunrock::util::HOST;
  CooT graph;
  graph.FromCsr(csr, target, 0, quiet, true);
  csr.Release();
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the gtf
  double elapsed_time = gunrock_gtf(parameters, graph, flow, maxflow);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
