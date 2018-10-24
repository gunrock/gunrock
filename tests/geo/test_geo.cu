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
  cudaError_t operator()(util::Parameters &parameters, VertexT v, SizeT s,
                         ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES | graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // <DONE> declare datastructures for reference result
    ValueT *ref_predicted_lat;
    ValueT *ref_predicted_lon;
    // </DONE>

    std::string labels_file = parameters.Get<std::string>("labels-file");

    util::PrintMsg("Labels File Input: " + labels_file, !quiet);

    // Input locations from a labels file
    ValueT *h_latitude = new ValueT[graph.nodes];
    ValueT *h_longitude = new ValueT[graph.nodes];

    retval =
        gunrock::graphio::labels::Read(parameters, h_latitude, h_longitude);

    util::PrintMsg("Debugging Labels -------------", !quiet);
    for (int p = 0; p < graph.nodes; p++) {
      util::PrintMsg("    locations[ " + std::to_string(p) + " ] = < " +
                         std::to_string(h_latitude[p]) + " , " +
                         std::to_string(h_longitude[p]) + " > ",
                     !quiet);
    }

    if (!quick) {
      // <DONE> init datastructures for reference result
      ref_predicted_lat = new ValueT[graph.nodes];
      ref_predicted_lon = new ValueT[graph.nodes];

      memcpy(ref_predicted_lat, h_latitude, graph.nodes * sizeof(ValueT));
      memcpy(ref_predicted_lon, h_longitude, graph.nodes * sizeof(ValueT));
      // </DONE>

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
        [
            // </DONE> pass necessary data to lambda
            h_latitude, h_longitude, ref_predicted_lat, ref_predicted_lon
            // </DONE>
    ](util::Parameters &parameters, GraphT &graph) {
          // <DONE> pass necessary data to app::Template::RunTests
          return app::geo::RunTests(parameters, graph, h_latitude, h_longitude,
                                    ref_predicted_lat, ref_predicted_lon,
                                    util::DEVICE);
          // </DONE>
        }));

    if (!quick) {
      // <DONE> deallocate host references
      delete[] ref_predicted_lat;
      ref_predicted_lat = NULL;
      delete[] ref_predicted_lon;
      ref_predicted_lon = NULL;
      // </DONE>
    }
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
