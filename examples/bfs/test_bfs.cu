// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <vector>
#include <gunrock/app/bfs/bfs_app.cu>
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
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_CSR | graph::HAS_CSC>
        GraphT;
    typedef typename GraphT::VertexT LabelT;
    typedef typename GraphT::CsrT CsrT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;  // graph we process on

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    GUARD_CU(app::Set_Srcs(parameters, graph));
    LabelT **ref_labels = NULL;
    int num_srcs = 0;
    bool quick = parameters.Get<bool>("quick");
    // compute reference CPU BFS solution for source-distance
    if (!quick) {
      bool quiet = parameters.Get<bool>("quiet");
      std::string validation = parameters.Get<std::string>("validation");
      util::PrintMsg("Computing reference value ...", !quiet);
      std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT> >("srcs");
      num_srcs = srcs.size();
      SizeT nodes = graph.nodes;
      ref_labels = new LabelT *[num_srcs];
      for (int i = 0; i < num_srcs; i++) {
        ref_labels[i] = (LabelT *)malloc(sizeof(LabelT) * nodes);
        auto src = srcs[i];
        VertexT num_iters = 0;
        util::PrintMsg("__________________________", !quiet);
        float elapsed = app::bfs::CPU_Reference(graph, src, quiet, false,
                                                ref_labels[i], NULL, num_iters);
        util::PrintMsg("--------------------------\nRun " + std::to_string(i) +
                           " elapsed: " + std::to_string(elapsed) +
                           " ms, src = " + std::to_string(src) +
                           ", #iterations = " + std::to_string(num_iters),
                       !quiet);
      }
    }

    std::vector<std::string> switches{
        "mark-pred", "advance-mode", "direction-optimized",
        "do-a",      "do-b",         "idempotence"};
    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [ref_labels](util::Parameters &parameters, GraphT &graph) {
          return app::bfs::RunTests(parameters, graph, ref_labels);
        }));

    if (!quick) {
      for (int i = 0; i < num_srcs; i++) {
        free(ref_labels[i]);
        ref_labels[i] = NULL;
      }
      delete[] ref_labels;
      ref_labels = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test bfs");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::bfs::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B | app::SIZET_U64B |
                           app::VALUET_U32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
