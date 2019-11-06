// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/sm/sm_app.cu>
#include <gunrock/app/test_base.cuh>

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
            typename ValueT>   // Use float as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES | graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT data_graph;
    GraphT pattern_graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, data_graph));
    GUARD_CU(graphio::LoadGraph(parameters, pattern_graph, "pattern-"));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");
    int num_runs = parameters.Get<int>("num-runs");

    util::PrintMsg(
        "# of nodes in data graph: " + std::to_string(data_graph.nodes),
        !quiet);
    util::PrintMsg(
        "# of nodes in query graph: " + std::to_string(pattern_graph.nodes),
        !quiet);

    // counts of matched subgraphs
    VertexT *ref_subgraph_match = new VertexT[data_graph.nodes];
    if (!quick) {
      util::PrintMsg("__________________________", !quiet);

      float elapsed =
          app::sm::CPU_Reference(parameters, data_graph.csr(),
                                 pattern_graph.csr(), ref_subgraph_match);

      util::PrintMsg("__________________________\nRun CPU Reference Avg. in " +
                         std::to_string(num_runs) + " iterations elapsed: " +
                         std::to_string(elapsed) + " ms",
                     !quiet);
    }

    std::vector<std::string> switches{"advance-mode"};
    GUARD_CU(app::Switch_Parameters(
        parameters, data_graph, switches,
        [&pattern_graph, &ref_subgraph_match](util::Parameters &parameters,
                                              GraphT &data_graph) {
          return app::sm::RunTests(parameters, data_graph, pattern_graph,
                                   ref_subgraph_match, util::DEVICE);
        }));

    if (ref_subgraph_match != NULL) {
      delete[] ref_subgraph_match;
      ref_subgraph_match = NULL;
    }
    GUARD_CU(pattern_graph.Release());
    GUARD_CU(data_graph.Release());
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test Subgraph Matching");
  GUARD_CU(graphio::UseParameters(parameters));
  // Recogonize pattern graph
  GUARD_CU(graphio::UseParameters(parameters, "pattern-"));
  GUARD_CU(app::sm::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B |  // app::VERTEXT_U64B |
                           app::SIZET_U32B |    // app::SIZET_U64B |
                           app::VALUET_F64B | app::UNDIRECTED | app::DIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
