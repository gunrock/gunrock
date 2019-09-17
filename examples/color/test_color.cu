// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_color.cu
 *
 * @brief Color driver program for Gunrock.
 */

#include <gunrock/app/color/color_app.cu>
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
                                    graph::HAS_EDGE_VALUES | graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;  // graph we process on

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // reference result on CPU:
    VertexT *ref_colors = NULL;

    bool quick = parameters.Get<bool>("quick");
    bool color_balance = parameters.Get<bool>("LBCOLOR");

    // compute reference CPU SSSP solution for source-distance
    if (!quick) {
      bool quiet = parameters.Get<bool>("quiet");
      std::string validation = parameters.Get<std::string>("validation");
      util::PrintMsg("Computing reference value ...", !quiet);

      // problem specific data:
      ref_colors = new VertexT[graph.nodes];

      util::PrintMsg("__________________________", !quiet);
      float elapsed =
          app::color::CPU_Reference(parameters, graph.csr(), ref_colors, quiet);

      util::PrintMsg(
          "--------------------------\n Elapsed: " + std::to_string(elapsed),
          !quiet);
    }

    std::vector<std::string> switches{"advance-mode"};
    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [color_balance, ref_colors](util::Parameters &parameters,
                                    GraphT &graph) {
          return app::color::RunTests(parameters, graph, color_balance,
                                      ref_colors, util::DEVICE);
        }));

    if (!quick) {
      delete[] ref_colors;
      ref_colors = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test color");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::color::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::SIZET_U32B |
                           app::VALUET_F32B | app::UNDIRECTED>(parameters,
                                                               main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
