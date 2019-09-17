// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bc.cu
 *
 * @brief Simple test driver program for Gunrock GC.
 */

#include <iostream>
#include <gunrock/app/bc/bc_app.cu>
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
    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // Enable is set sources
    GUARD_CU(app::Set_Srcs(parameters, graph));
    int num_srcs = 0;

    ValueT **reference_bc_values = NULL;
    ValueT **reference_sigmas = NULL;
    VertexT **reference_source_path = NULL;

    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    if (!quick) {
      // std::string validation = parameters.Get<std::string>("validation");
      util::PrintMsg("Computing reference value ...", !quiet);
      std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT> >("srcs");
      num_srcs = srcs.size();

      SizeT nodes = graph.nodes;

      reference_bc_values = new ValueT *[num_srcs];
      reference_sigmas = new ValueT *[num_srcs];
      reference_source_path = new VertexT *[num_srcs];

      for (int i = 0; i < num_srcs; i++) {
        VertexT src = srcs[i];
        util::PrintMsg("__________________________", !quiet);

        reference_bc_values[i] = new ValueT[nodes];
        reference_sigmas[i] = new ValueT[nodes];
        reference_source_path[i] = new VertexT[nodes];

        float elapsed = app::bc::CPU_Reference(
            graph, reference_bc_values[i], reference_sigmas[i],
            reference_source_path[i], src, quiet);
        util::PrintMsg("--------------------------\nRun " + std::to_string(i) +
                           " elapsed: " + std::to_string(elapsed) +
                           " ms, src = " + std::to_string(src),
                       !quiet);
      }
    }

    std::vector<std::string> switches{"advance-mode"};
    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [reference_bc_values, reference_sigmas, reference_source_path](
            util::Parameters &parameters, GraphT &graph) {
          return app::bc::RunTests(parameters, graph, reference_bc_values,
                                   reference_sigmas, reference_source_path);
        }));

    // Cleanup
    if (!quick) {
      for (int i = 0; i < num_srcs; i++) {
        delete[] reference_bc_values[i];
        reference_bc_values[i] = NULL;
        delete[] reference_sigmas[i];
        reference_sigmas[i] = NULL;
        delete[] reference_source_path[i];
        reference_source_path[i] = NULL;
      }
      delete[] reference_bc_values;
      reference_bc_values = NULL;
      delete[] reference_sigmas;
      reference_sigmas = NULL;
      delete[] reference_source_path;
      reference_source_path = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test bc");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::bc::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B |  // app::VERTEXT_U64B |
                           app::SIZET_U32B |    // app::SIZET_U64B |
                           app::VALUET_F32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
