// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_knn.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/knn/knn_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::knn;

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
  cudaError_t operator()(util::Parameters& parameters, VertexT v, SizeT s,
                         ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    // Get number of nearest neighbors, default k = 10
    SizeT k = parameters.Get<int>("k");
    // Get x reference point, default point_id = 0
    VertexT point_x = parameters.Get<VertexT>("x");
    // Get y reference point, default point_id = 0
    VertexT point_y = parameters.Get<VertexT>("y");
    // Get number of neighbors two close points should share
    SizeT eps = parameters.Get<SizeT>("eps");
    // Get the min density
    SizeT min_pts = parameters.Get<SizeT>("min-pts");

    util::PrintMsg("Reference point is (" + std::to_string(point_x) + ", " +
                       std::to_string(point_y) + "), k = " + std::to_string(k) +
                       ", eps = " + std::to_string(eps) +
                       +", min-pts = " + std::to_string(min_pts) + "\n",
                   !quiet);

    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    //util::CpuTimer cpu_timer;
    GraphT graph;

    //cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    //cpu_timer.Stop();
    //parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // <TODO> get srcs if needed, e.g.:
    // GUARD_CU(app::Set_Srcs (parameters, graph));
    // std::vector<VertexT> srcs
    //    = parameters.Get<std::vector<VertexT> >("srcs");
    // int num_srcs = srcs.size();
    // </TODO>

    // Reference result on CPU
    SizeT* ref_k_nearest_neighbors = NULL;
    SizeT* h_knns = NULL;
    h_knns = new SizeT[k];

    if (!quick) {
      // Init datastructures for reference result on GPU
      ref_k_nearest_neighbors = new SizeT[k];

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);
      util::PrintMsg("______ CPU Reference _____", !quiet);

      float elapsed = app::knn::CPU_Reference(graph.csr(), k, point_x, point_y,
                                              ref_k_nearest_neighbors, quiet);

      util::PrintMsg(
          "--------------------------\n Elapsed: " + std::to_string(elapsed),
          !quiet);
      util::PrintMsg("__________________________", !quiet);
    }

    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [k, eps, min_pts, h_knns, ref_k_nearest_neighbors](
            util::Parameters& parameters, GraphT& graph) {
          return app::knn::RunTests(parameters, graph, k, eps, min_pts, h_knns,
                                    ref_k_nearest_neighbors, util::DEVICE);
        }));

    if (!quick) {
      delete[] ref_k_nearest_neighbors;
      ref_k_nearest_neighbors = NULL;
    }

    delete[] h_knns;
    h_knns = NULL;

    return retval;
  }
};

int main(int argc, char** argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test knn");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::knn::UseParameters(parameters));
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
