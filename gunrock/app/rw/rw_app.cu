// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file rw_app.cu
 *
 * @brief Simple Gunrock Application
 */

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

#include <gunrock/app/rw/rw_enactor.cuh>
#include <gunrock/app/rw/rw_test.cuh>

namespace gunrock {
namespace app {
namespace rw {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<int>(
      "walk-length", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 10,
      "length of random walks", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "walk-mode", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0,
      "random walk mode (0=uniform_random; 1=greedy, 2=stochastic_greedy)",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "store-walks", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, true,
      "store random walks?", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "node-value-path", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, "",
      "path to file containing node values", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "walks-per-node", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 1,
      "number of random walks per source node", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "seed", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, time(NULL),
      "seed for random number generator", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run RW tests
 * @tparam     GraphT           Type of the graph
 * @tparam     ValueT           Type of the distances
 * @param[in]  parameters       Excution parameters
 * @param[in]  graph            Input graph
 * @param[in]  walk_length      Length of random walks
 * @param[in]  walks_per_node   Number of random walks per node
 * @param[in]  ref_walks        Array of random walks from CPU
 * @param[in]  target           where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     int walk_length, int walks_per_node, int walk_mode,
                     bool store_walks, typename GraphT::VertexT *ref_walks,
                     util::Location target) {
  cudaError_t retval = cudaSuccess;

  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");
  int num_runs = parameters.Get<int>("num-runs");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("rw", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem specific host data
  VertexT *h_walks = NULL;
  if (store_walks) {
    h_walks = new VertexT[graph.nodes * walk_length * walks_per_node];
  }

  uint64_t *h_neighbors_seen = new uint64_t[graph.nodes * walks_per_node];
  uint64_t *h_steps_taken = new uint64_t[graph.nodes * walks_per_node];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(target));
    GUARD_CU(enactor.Reset(walks_per_node, target));

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
      GUARD_CU(problem.Extract(h_walks, h_neighbors_seen, h_steps_taken));
      SizeT num_errors = Validate_Results(
          parameters, graph, walk_length, walks_per_node, walk_mode,
          store_walks, h_walks, h_neighbors_seen, h_steps_taken, ref_walks);
    }
  }

  cpu_timer.Start();

  if (validation == "last") {
    GUARD_CU(problem.Extract(h_walks, h_neighbors_seen, h_steps_taken));
    SizeT num_errors = Validate_Results(
        parameters, graph, walk_length, walks_per_node, walk_mode, store_walks,
        h_walks, h_neighbors_seen, h_steps_taken, ref_walks);
  }

  // compute running statistics
  // TODO: change NULL to problem specific per-vertex visited marker, e.g.
  // h_distances
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
// Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  delete[] h_walks;
  h_walks = NULL;
  delete[] h_neighbors_seen;
  h_neighbors_seen = NULL;
  delete[] h_steps_taken;
  h_steps_taken = NULL;
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace rw
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_rw function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the distances
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] distances  Return shortest distance to source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_rw(gunrock::util::Parameters &parameters, GraphT &graph,
                  typename GraphT::VertexT *h_walks, int walks_per_node) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::rw::Problem<GraphT> ProblemT;
  typedef gunrock::app::rw::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, target);
  enactor.Init(problem, target);

  int num_runs = parameters.Get<int>("num-runs");
  for (int run_num = 0; run_num < num_runs; ++run_num) {
    problem.Reset(target);
    enactor.Reset(walks_per_node, target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(h_walks);
  }

  enactor.Release(target);
  problem.Release(target);
  return total_time;
}

//  * @brief Simple interface take in graph as CSR format
//  * @param[in]  num_nodes   Number of veritces in the input graph
//  * @param[in]  num_edges   Number of edges in the input graph
//  * @param[in]  row_offsets CSR-formatted graph input row offsets
//  * @param[in]  col_indices CSR-formatted graph input column indices
//  * @param[in]  edge_values CSR-formatted graph input edge weights
//  * @param[in]  num_runs    Number of runs to perform SSSP
//  * @param[in]  walks          Array for random walks
//  * @param[in]  walks_per_node Number of random walks per node
//  * \return     double      Return accumulated elapsed times for all runs

template <typename VertexT = int, typename SizeT = int,
          typename GValueT = unsigned int, typename TValueT = GValueT>
float rw(const SizeT num_nodes, const SizeT num_edges, const SizeT *row_offsets,
         const VertexT *col_indices, const int num_runs, VertexT *h_walks,
         const int walks_per_node) {
  // TODO: change to other graph representation, if not using CSR
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("rw");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::rw::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");
  parameters.Set("num-runs", num_runs);

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;

  graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, num_nodes + 1,
                                     gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, num_edges,
                                        gunrock::util::HOST);
  graph.FromCsr(graph.csr(), true, quiet);
  gunrock::graphio::LoadGraph(parameters, graph);

  double elapsed_time = gunrock_rw(parameters, graph, h_walks, walks_per_node);

  graph.Release();

  return elapsed_time;
}

float rw(const int num_nodes, const int num_edges, const int *row_offsets,
         const int *col_indices, const int num_runs, int *h_walks,
         const int walks_per_node) {
  return rw(num_nodes, num_edges, row_offsets, col_indices, num_runs, h_walks,
            walks_per_node);
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
