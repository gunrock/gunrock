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

#include <gunrock/app/snn/snn_app.cu>
#include <gunrock/app/test_base.cuh>

// JSON includes
#include <gunrock/util/info_rapidjson.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::snn;

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

    if (min_pts >= k) {
      util::PrintMsg("Min pts should be smaller than k", true);
      return (cudaError_t)1;
    }

    util::PrintMsg("Reference point is (" + std::to_string(point_x) + ", " +
                       std::to_string(point_y) + "), k = " + std::to_string(k) +
                       ", eps = " + std::to_string(eps) +
                       +", min-pts = " + std::to_string(min_pts) + "\n",
                   !quiet);

    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // Reference result on CPU
    SizeT* ref_cluster = NULL;

    SizeT* h_cluster = (SizeT*)malloc(sizeof(SizeT) * graph.nodes);

    SizeT* ref_core_point_counter = NULL;
    SizeT* h_core_point_counter = (SizeT*)malloc(sizeof(SizeT));

    SizeT* ref_cluster_counter = NULL;
    SizeT* h_cluster_counter = (SizeT*)malloc(sizeof(SizeT));

    //SizeT* ref_knns = NULL;
    //SizeT* h_knns = (SizeT*)malloc(sizeof(SizeT) * graph.nodes * k);

    if (!quick) {
      // Init datastructures for reference result on GPU
      ref_cluster = (SizeT*)malloc(sizeof(SizeT) * graph.nodes);
      for (auto i = 0; i < graph.nodes; ++i) ref_cluster[i] = i;

      ref_core_point_counter = (SizeT*)malloc(sizeof(SizeT));
      ref_cluster_counter = (SizeT*)malloc(sizeof(SizeT));
      //ref_knns = (SizeT*)malloc(sizeof(SizeT) * graph.nodes * k);

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);
      util::PrintMsg("______ CPU Reference _____", !quiet);

      float elapsed =
          app::snn::CPU_Reference(graph.csr(), k, eps, min_pts, point_x,
                                  point_y, /*ref_knns,*/ ref_cluster, 
                                  ref_core_point_counter, ref_cluster_counter,
                                  quiet);

      util::PrintMsg(
          "--------------------------\n Elapsed: " + std::to_string(elapsed),
          !quiet);
      util::PrintMsg("__________________________", !quiet);
      parameters.Set("cpu-elapsed", elapsed);
    }

    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [k, eps, min_pts, /*h_knns, ref_knns, */h_cluster, h_core_point_counter,
         h_cluster_counter, ref_core_point_counter, ref_cluster_counter,
         ref_cluster](util::Parameters& parameters, GraphT& graph) {
          return app::snn::RunTests(parameters, graph, k, eps, min_pts, /*h_knns,
                                    ref_knns,*/ h_cluster, ref_cluster,
                                    h_core_point_counter, ref_core_point_counter, 
                                    h_cluster_counter, ref_cluster_counter,
                                    util::DEVICE);
        }));

    if (!quick) {
      delete[] ref_cluster;
      //delete[] ref_knns;
      delete[] ref_core_point_counter;
      delete[] ref_cluster_counter;
    }

    delete[] h_cluster;

    return retval;
  }
};

int main(int argc, char** argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test knn");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::snn::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B | app::SIZET_U64B |
                           app::VALUET_S64B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
