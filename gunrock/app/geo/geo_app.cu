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
  util::Location memspace = util::HOST;

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

  GUARD_CU(problem.Init(graph, memspace, target));

  util::PrintMsg("Initializing enactor ... ", !quiet_mode);

  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(h_latitude.GetPointer(util::HOST),
                           h_longitude.GetPointer(util::HOST), geo_iter,
                           spatial_iter, target, memspace));
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
      GUARD_CU(problem.Extract(h_predicted_lat, h_predicted_lon, target, memspace));

      SizeT num_errors =
          Validate_Results(parameters, graph, h_predicted_lat, h_predicted_lon,
                           ref_predicted_lat, ref_predicted_lon, false);
    }
  }

  cpu_timer.Start();

  // Extract problem data
  GUARD_CU(problem.Extract(h_predicted_lat, h_predicted_lon, target, memspace));

  if (validation == "last") {
    SizeT num_errors =
        Validate_Results(parameters, graph, h_predicted_lat, h_predicted_lon,
                         ref_predicted_lat, ref_predicted_lon, false);
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

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace geo
}  // namespace app
}  // namespace gunrock

/*
 * @brief Entry of gunrock_geo function
 * @tparam     GraphT     Type of the graph
 * @tparam     VertexT    Type of the vertices
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] longitudes Return predicted longitudes
 * @param[out] latitudes  Return predicted latitudes
 * @param[in]  memspace   Location of allocated arrays (default = HOST)
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_geo(gunrock::util::Parameters &parameters,
                     GraphT &graph,
                     ValueT *latitudes,
                     ValueT *longitudes,
                     ValueT *predicted_lat,
                     ValueT *predicted_long,
                     gunrock::util::Location memspace = gunrock::util::HOST) {
  typedef gunrock::app::geo::Problem<GraphT> ProblemT;
  typedef gunrock::app::geo::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  int geo_iter = parameters.Get<int>("geo-iter");
  int spatial_iter = parameters.Get<int>("spatial-iter");

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;

  problem.Init(graph, memspace, target);
  enactor.Init(problem, target);

  problem.Reset(latitudes, longitudes, geo_iter, spatial_iter, target, memspace);
  enactor.Reset(target);

  cpu_timer.Start();
  enactor.Enact();
  cpu_timer.Stop();

  total_time += cpu_timer.ElapsedMillis();

  // Extract problem data
  problem.Extract(predicted_lat, predicted_long, target, memspace);

  // Clean up
  enactor.Release(target);
  problem.Release(target);

  return total_time;
}

/*
 * @brief Entry of gunrock_geo function
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename VertexT, typename SizeT,
          typename GValueT>
double geo(const SizeT num_nodes, const SizeT num_edges,
           const SizeT *row_offsets, const VertexT *col_indices,
           GValueT *latitudes, GValueT *longitudes,
           GValueT *predicted_lat, GValueT* predicted_long,
           gunrock::util::Location memspace = gunrock::util::HOST) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
                                           gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
  gunrock::util::Parameters parameters("geo");
  gunrock::graphio::UseParameters(parameters);
  gunrock::app::geo::UseParameters(parameters);
  gunrock::app::UseParameters_test(parameters);
  parameters.Parse_CommandLine(0, NULL);
  parameters.Set("graph-type", "by-pass");

  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(num_nodes, num_edges, memspace);
  graph.CsrT::row_offsets.SetPointer((SizeT *)row_offsets, num_nodes + 1,
                                     memspace);
  graph.CsrT::column_indices.SetPointer((VertexT *)col_indices, num_edges,
                                        memspace);
  graph.FromCsr(graph.csr(), memspace, 0, quiet, true);

  // Run the geolocation
  double elapsed_time = gunrock_geo(parameters, graph, latitudes, longitudes, predicted_lat, predicted_long, memspace);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

double geo(const int num_nodes, const int num_edges, const int *row_offsets,
             const int *col_indices, float *latitudes, float* longitudes, float* predicted_lat, float* predicted_long) {
  return geo<int, int, float>(num_nodes, num_edges, row_offsets, col_indices, latitudes, longitudes, predicted_lat, predicted_long);
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// // End:
