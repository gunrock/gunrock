// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_hits.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/hits/hits_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::hits;

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
    int max_iter = parameters.Get<SizeT>("max-iter");

    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_CSR | graph::HAS_COO>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // Declare datastructures for reference result on GPU
    ValueT *ref_hrank;
    ValueT *ref_arank;

    if (!quick) {
      // Init datastructures for reference result on GPU
      ref_hrank = new ValueT[graph.nodes];
      ref_arank = new ValueT[graph.nodes];

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);

      float elapsed = app::hits::CPU_Reference(graph.coo(), ref_hrank,
                                               ref_arank, max_iter, quiet);

      util::PrintMsg("--------------------------\n CPU Elapsed: " +
                         std::to_string(elapsed),
                     !quiet);
    }

    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [ref_hrank, ref_arank](util::Parameters &parameters, GraphT &graph) {
          return app::hits::RunTests(parameters, graph, ref_hrank, ref_arank,
                                     util::DEVICE);
        }));

    if (!quick) {
      // Deallocate host references
      delete[] ref_hrank;
      ref_hrank = NULL;
      delete[] ref_arank;
      ref_arank = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test hits");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::hits::UseParameters(parameters));
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
