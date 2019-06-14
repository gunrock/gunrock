// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cu
 *
 * @brief Simple test driver program for PageRank.
 */

#include <gunrock/app/pr/pr_app.cu>
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
                                    graph::HAS_COO | graph::HAS_CSC>
        GraphT;
    // typedef typename GraphT::CooT CooT;

    cudaError_t retval = cudaSuccess;
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    util::CpuTimer cpu_timer;
    GraphT graph;  // graph we process on
    // require undirected input graph when unnormalized
    if (!parameters.Get<bool>("normalize")) {
      util::PrintMsg(
          "Directed graph is only supported by normalized PR,"
          " Changing graph type to undirected.",
          quiet);
      parameters.Set("undirected", true);
    }

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    std::vector<bool> compensate_vec =
        parameters.Get<std::vector<bool>>("compensate");
    for (auto it = compensate_vec.begin(); it != compensate_vec.end(); it++) {
      bool compensate = *it;
      if (compensate) {
        GUARD_CU(gunrock::app::pr::Compensate_ZeroDegrees(graph, quiet));
      }
      GUARD_CU(parameters.Set("compensate", compensate));
      std::vector<std::string> switches{"normalize", "delta", "threshold",
                                        "max-iter", "pull"};
      GUARD_CU(app::Switch_Parameters(
          parameters, graph, switches,
          [quick, quiet](util::Parameters &parameters, GraphT &graph) {
            cudaError_t retval = cudaSuccess;
            GUARD_CU(app::Set_Srcs(parameters, graph));
            ValueT **ref_ranks = NULL;
            VertexT **ref_vertices = NULL;
            int num_srcs = 0;

            // compute reference CPU SSSP solution for source-distance
            if (!quick) {
              util::PrintMsg("Computing reference value ...", !quiet);
              std::vector<VertexT> srcs =
                  parameters.Get<std::vector<VertexT>>("srcs");
              num_srcs = srcs.size();
              SizeT nodes = graph.nodes;
              ref_ranks = (ValueT **)malloc(sizeof(ValueT *) * num_srcs);
              ref_vertices = (VertexT **)malloc(sizeof(VertexT *) * num_srcs);
              for (int i = 0; i < num_srcs; i++) {
                ref_ranks[i] = (ValueT *)malloc(sizeof(ValueT) * nodes);
                ref_vertices[i] = (VertexT *)malloc(sizeof(VertexT) * nodes);
                VertexT src = srcs[i];
                util::PrintMsg("__________________________", !quiet);
                float elapsed = app::pr::CPU_Reference(
                    parameters, graph, src, ref_vertices[i], ref_ranks[i]);
                util::PrintMsg("--------------------------\nRun " +
                                   std::to_string(i) +
                                   " elapsed: " + std::to_string(elapsed) +
                                   " ms, src = " + std::to_string(src),
                               !quiet);
              }
            }

            std::vector<std::string> switches2{"scale", "advance-mode"};
            GUARD_CU(app::Switch_Parameters(
                parameters, graph, switches2,
                [ref_ranks, ref_vertices](util::Parameters &parameters,
                                          GraphT &graph) {
                  return app::pr::RunTests(parameters, graph, ref_vertices,
                                           ref_ranks);
                }));

            if (!quick) {
              for (int i = 0; i < num_srcs; i++) {
                free(ref_ranks[i]);
                ref_ranks[i] = NULL;
                free(ref_vertices[i]);
                ref_vertices[i] = NULL;
              }
              free(ref_ranks);
              ref_ranks = NULL;
              free(ref_vertices);
              ref_vertices = NULL;
            }
            return retval;
          }));
    }
    GUARD_CU(parameters.Set("compensate", compensate_vec));
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test pr");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::pr::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B |  // app::VERTEXT_U64B |
                           app::SIZET_U32B |    // app::SIZET_U64B |
                           app::VALUET_F64B |   // app::VALUET_F64B |
                           app::DIRECTED | app::UNDIRECTED>(parameters,
                                                            main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
