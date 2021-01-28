// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_dynamic_graph.cu
 *
 * @brief Simple test driver program for dynamic graph building.
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph defintions
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>

#include <gunrock/app/test_base.cuh>

#include <examples/dynamic_graph/validation.cuh>

using namespace gunrock;

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
    using WeightedCooCsrT = app::TestGraph<VertexT, SizeT, ValueT,
                                           graph::HAS_EDGE_VALUES |
                                               graph::HAS_COO | graph::HAS_CSR>;
    using WeightedDynT =
        app::TestGraph<VertexT, SizeT, ValueT,
                       graph::HAS_EDGE_VALUES | graph::HAS_DYN>;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer, total_timer;
    total_timer.Start();

    float load_factor = parameters.Get<float>("load-factor");
    bool quick = parameters.Get<bool>("quick");
    std::string validation = parameters.Get<std::string>("validation");

    WeightedCooCsrT input_graph;
    auto &input_graph_csr = input_graph.csr();
    auto &input_graph_coo = input_graph.coo();

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, input_graph));
    // Sort once for validation
    input_graph_csr.Sort();
    cpu_timer.Stop();

    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // Move the graph to the GPU
    input_graph.Move(util::HOST, util::DEVICE);

    util::Info info("DynamicGraphBuilding", parameters,
                    input_graph);  // initialize Info structure

    int num_runs = parameters.Get<int>("num-runs");
    bool quiet = parameters.Get<bool>("quiet");
    double total_time = 0;

    SizeT nodes = input_graph.nodes;
    SizeT edges = input_graph.edges;
    for (int run_num = 0; run_num < num_runs; ++run_num) {
      WeightedDynT result_graph;

      auto &result_dyn_graph = result_graph.dyn();

      result_dyn_graph.Allocate(nodes);
      result_dyn_graph.dynamicGraph.InitHashTables(
          nodes, load_factor,
          input_graph_csr.row_offsets.GetPointer(util::HOST));
      result_dyn_graph.is_directed = true;
      result_dyn_graph.nodes = nodes;
      result_dyn_graph.edges = edges;
      bool directed_edges = false;  // Input is from Market loader which will
                                    // always double the edges
      util::PrintMsg("__________________________", !quiet);

      gunrock::util::GpuTimer gpu_timer;
      gpu_timer.Start();
      result_dyn_graph.InsertEdgesBatch(input_graph_coo.edge_pairs,
                                        input_graph_coo.edge_values, edges,
                                        directed_edges, util::DEVICE);
      gpu_timer.Stop();
      info.CollectSingleRun(gpu_timer.ElapsedMillis());

      util::PrintMsg(
          "--------------------------\nRun " + std::to_string(run_num) +
              " elapsed: " + std::to_string(gpu_timer.ElapsedMillis()) + " ms.",
          !quiet);

      if (!quick && (validation == "each" ||
          validation == "last" && run_num == num_runs - 1)) {
        bool failed =
            CompareWeightedDynCSR(input_graph_csr, result_dyn_graph, quiet);
        if (failed) {
          util::PrintMsg("FAILED", !quiet);
          std::exit(EXIT_FAILURE);
        }
      }
      result_dyn_graph.Release();
    }

    util::PrintMsg("PASSED", !quiet);
    cpu_timer.Start();

    input_graph.Release(util::HOST | util::DEVICE);

    cpu_timer.Stop();
    total_timer.Stop();
    info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());

    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test dynamic graph bulk-build");
  parameters.Use<float>(
      "load-factor",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.7, "Global Load Factor.", __FILE__, __LINE__);

  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(app::UseParameters_app(parameters));

  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::SIZET_U32B |
                           app::VALUET_S32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
