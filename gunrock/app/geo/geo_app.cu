// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file geo_app.cu
 *
 * @brief Geolocation Application
 */

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/graphio/graphio.cuh>
#include <gunrock/graphio/labels.cuh>

#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

#include <gunrock/app/geo/geo_enactor.cuh>
#include <gunrock/app/geo/geo_test.cuh>

namespace gunrock {
namespace app {
namespace geo {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<int>(
      "geo-iter",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      3, "Number of iterations geolocation should run for (default=3).",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "spatial-iter",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1000,
      "Number of maximum iterations spatial median "
      "kernel should run for (default=1000).",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "geo-complete",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false,
      "Run geolocation application until all locations for all nodes are "
      "found, uses an atomic (default=false).",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "labels-file",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", "User locations label file for geolocation app.", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "debug",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false,
      "Debug label values, this prints out the entire labels array (longitude, "
      "latitude).",
      __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run geolocation tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  target        where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ArrayT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     ArrayT &h_latitude, ArrayT &h_longitude,
                     ArrayT &ref_predicted_lat, ArrayT &ref_predicted_lon,
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

  int geo_iter = parameters.Get<int>("geo-iter");
  int spatial_iter = parameters.Get<int>("spatial-iter");

  util::PrintMsg("Number of iterations: " + std::to_string(geo_iter),
                 !quiet_mode);

  util::Info info("geolocation", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem specific host data array to
  // extract device values to host
  ValueT *h_predicted_lat = new ValueT[graph.nodes];
  ValueT *h_predicted_lon = new ValueT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;

  util::PrintMsg("Initializing problem ... ", !quiet_mode);

  GUARD_CU(problem.Init(graph, target));

  util::PrintMsg("Initializing enactor ... ", !quiet_mode);

  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(h_latitude.GetPointer(util::HOST),
                           h_longitude.GetPointer(util::HOST), geo_iter,
                           spatial_iter, target));
    GUARD_CU(enactor.Reset(target));

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
      GUARD_CU(problem.Extract(h_predicted_lat, h_predicted_lon));

      SizeT num_errors =
          Validate_Results(parameters, graph, h_predicted_lat, h_predicted_lon,
                           ref_predicted_lat, ref_predicted_lon, false);
    }
  }

  cpu_timer.Start();

  // Extract problem data
  GUARD_CU(problem.Extract(h_predicted_lat, h_predicted_lon));

  if (validation == "last") {
    SizeT num_errors =
        Validate_Results(parameters, graph, h_predicted_lat, h_predicted_lon,
                         ref_predicted_lat, ref_predicted_lon, false);
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

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace geo
}  // namespace app
}  // namespace gunrock

// ===========================================================================================
// ========================= CODE BELOW THIS LINE NOT NEEDED FOR TESTS
// =======================
// ===========================================================================================

// /*
// * @brief Entry of gunrock_template function
// * @tparam     GraphT     Type of the graph
// * @tparam     ValueT     Type of the distances
// * @param[in]  parameters Excution parameters
// * @param[in]  graph      Input graph
// * @param[out] distances  Return shortest distance to source per vertex
// * @param[out] preds      Return predecessors of each vertex
// * \return     double     Return accumulated elapsed times for all runs
// */
// template <typename GraphT, typename ValueT = typename GraphT::ValueT>
// double gunrock_Template(
//     gunrock::util::Parameters &parameters,
//     GraphT &graph
//     // TODO: add problem specific outputs, e.g.:
//     //ValueT **distances
//     )
// {
//     typedef typename GraphT::VertexT VertexT;
//     typedef gunrock::app::Template::Problem<GraphT  > ProblemT;
//     typedef gunrock::app::Template::Enactor<ProblemT> EnactorT;
//     gunrock::util::CpuTimer cpu_timer;
//     gunrock::util::Location target = gunrock::util::DEVICE;
//     double total_time = 0;
//     if (parameters.UseDefault("quiet"))
//         parameters.Set("quiet", true);

//     // Allocate problem and enactor on GPU, and initialize them
//     ProblemT problem(parameters);
//     EnactorT enactor;
//     problem.Init(graph  , target);
//     enactor.Init(problem, target);

//     int num_runs = parameters.Get<int>("num-runs");
//     // TODO: get problem specific inputs, e.g.:
//     // std::vector<VertexT> srcs =
//     parameters.Get<std::vector<VertexT>>("srcs");
//     // int num_srcs = srcs.size();
//     for (int run_num = 0; run_num < num_runs; ++run_num)
//     {
//         // TODO: problem specific inputs, e.g.:
//         // int src_num = run_num % num_srcs;
//         // VertexT src = srcs[src_num];
//         problem.Reset(/*src,*/ target);
//         enactor.Reset(/*src,*/ target);

//         cpu_timer.Start();
//         enactor.Enact(/*src*/);
//         cpu_timer.Stop();

//         total_time += cpu_timer.ElapsedMillis();
//         // TODO: extract problem specific data, e.g.:
//         problem.Extract(/*distances[src_num]*/);
//     }

//     enactor.Release(target);
//     problem.Release(target);
//     // TODO: problem specific clean ups, e.g.:
//     // srcs.clear();
//     return total_time;
// }

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

// template <
//     typename VertexT = int,
//     typename SizeT   = int,
//     typename GValueT = unsigned int,
//     typename TValueT = GValueT>
// float Geolocation(
//     const SizeT        num_nodes,
//     const SizeT        num_edges,
//     const SizeT       *row_offsets,
//     const VertexT     *col_indices,
//     const GValueT     *edge_values,
//     const int          num_runs
//     // TODO: add problem specific inputs and outputs, e.g.:
//     //      VertexT     *sources,
//     //      SSSPValueT **distances
//     )
// {
//     // TODO: change to other graph representation, if not using CSR
//     typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
//         gunrock::graph::HAS_EDGE_VALUES | gunrock::graph::HAS_CSR>
//         GraphT;
//     typedef typename GraphT::CsrT CsrT;

//     // Setup parameters
//     gunrock::util::Parameters parameters("Template");
//     gunrock::graphio::UseParameters(parameters);
//     gunrock::app::Template::UseParameters(parameters);
//     gunrock::app::UseParameters_test(parameters);
//     parameters.Parse_CommandLine(0, NULL);
//     parameters.Set("graph-type", "by-pass");
//     parameters.Set("num-runs", num_runs);
//     // TODO: problem specific inputs, e.g.:
//     // std::vector<VertexT> srcs;
//     // for (int i = 0; i < num_runs; i ++)
//     //     srcs.push_back(sources[i]);
//     // parameters.Set("srcs", srcs);

//     bool quiet = parameters.Get<bool>("quiet");
//     GraphT graph;
//     // Assign pointers into gunrock graph format
//     // TODO: change to other graph representation, if not using CSR
//     graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
//     graph.CsrT::row_offsets   .SetPointer(row_offsets, num_nodes + 1,
//     gunrock::util::HOST); graph.CsrT::column_indices.SetPointer(col_indices,
//     num_edges, gunrock::util::HOST); graph.FromCsr(graph.csr(), true, quiet);
//     gunrock::graphio::LoadGraph(parameters, graph);

//     // Run the Template
//     // TODO: add problem specific outputs, e.g.
//     double elapsed_time = gunrock_Template(parameters, graph /*,
//     distances*/);

//     // Cleanup
//     graph.Release();
//     // TODO: problem specific cleanup
//     // srcs.clear();

//     return elapsed_time;
// }

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// // End:
