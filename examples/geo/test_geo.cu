// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_hello.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/geo/geo_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::geo;

/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT    Type of vertex identifier
   * @tparam SizeT      Type of graph size, i.e. type of edge identifier
   * @tparam ValueT     Type of edge values
   * @param  parameters Command line parameters
   * @param  v,s,val    Place holders for type deduction
   * \return cudaError_t error message(s), if any
   */
  template <typename VertexT,  // Use int as the vertex identifier
            typename SizeT,    // Use int as the graph size type
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");
    bool debug = parameters.Get<bool>("debug");

    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES | graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));

    std::string labels_file = parameters.Get<std::string>("labels-file");
    util::PrintMsg("Labels File Input: " + labels_file, !quiet);

    // Input locations from a labels file
    util::Array1D<SizeT, ValueT> h_latitude;
    util::Array1D<SizeT, ValueT> h_longitude;

    h_latitude.SetName("h_latitude");
    h_longitude.SetName("h_longitude");

    GUARD_CU(h_latitude.Allocate(graph.nodes, util::HOST));
    GUARD_CU(h_longitude.Allocate(graph.nodes, util::HOST));

    retval =
        gunrock::graphio::labels::Read(parameters, h_latitude, h_longitude);

    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    if (!quiet && debug) {
      util::PrintMsg("Debugging Labels -------------", !quiet);
      for (int p = 0; p < graph.nodes; p++) {
        util::PrintMsg("    locations[ " + std::to_string(p) + " ] = < " +
                           std::to_string(h_latitude[p]) + " , " +
                           std::to_string(h_longitude[p]) + " > ",
                       !quiet);
      }
    }

    util::Array1D<SizeT, ValueT> ref_predicted_lat;
    util::Array1D<SizeT, ValueT> ref_predicted_lon;

    ref_predicted_lat.SetName("ref_predicted_lat");
    ref_predicted_lon.SetName("ref_predicted_lon");

    if (!quick) {
      // init datastructures for reference result
      GUARD_CU(ref_predicted_lat.Allocate(graph.nodes, util::HOST));
      GUARD_CU(ref_predicted_lon.Allocate(graph.nodes, util::HOST));

      GUARD_CU(ref_predicted_lat.SetPointer(h_latitude.GetPointer(util::HOST),
                                            graph.nodes, util::HOST));
      GUARD_CU(ref_predicted_lat.Move(util::HOST, util::HOST));

      GUARD_CU(ref_predicted_lon.SetPointer(h_longitude.GetPointer(util::HOST),
                                            graph.nodes, util::HOST));
      GUARD_CU(ref_predicted_lon.Move(util::HOST, util::HOST));

      bool geo_complete = parameters.Get<bool>("geo-complete");
      int geo_iter = parameters.Get<int>("geo-iter");
      int spatial_iter = parameters.Get<int>("spatial-iter");

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);
      util::PrintMsg("______ CPU Reference _____", !quiet);

      float elapsed = app::geo::CPU_Reference(
          graph.csr(), ref_predicted_lat, ref_predicted_lon, geo_iter,
          spatial_iter, geo_complete, quiet);

      util::PrintMsg(
          "--------------------------\n Elapsed: " + std::to_string(elapsed),
          !quiet);

      util::PrintMsg("__________________________", !quiet);
    }

    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [h_latitude, h_longitude, ref_predicted_lat, ref_predicted_lon](
            util::Parameters &parameters, GraphT &graph) {
          return app::geo::RunTests(parameters, graph, h_latitude, h_longitude,
                                    ref_predicted_lat, ref_predicted_lon,
                                    util::DEVICE);
        }));

    if (!quick) {
      GUARD_CU(ref_predicted_lat.Release());
      GUARD_CU(ref_predicted_lon.Release());
    }

    GUARD_CU(h_latitude.Release());
    GUARD_CU(h_longitude.Release());

    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test geolocation");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::geo::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B | app::SIZET_U64B |
                           app::VALUET_F32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
