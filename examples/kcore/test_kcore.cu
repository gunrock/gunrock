// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_kcore.cu
 *
 * @brief Simple test driver program for vertex k-core decomposition.
 */

#include <gunrock/app/kcore/kcore_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::kcore;

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
    parameters.Set("undirected", true);

    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                        graph::HAS_COO |
                                        graph::HAS_CSR | graph::HAS_CSC>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    //Declare data structures for reference result on CPU
    SizeT *ref_k_cores = NULL;

    if (!quick) {
      //Init data structures for reference result on CPU
      ref_k_cores = new SizeT[graph.nodes];

      // If not in `quick` mode, compute CPU reference implementation
      util::PrintMsg("__________________________", !quiet);

      float elapsed =
          app::kcore::CPU_Reference(graph.csr(), ref_k_cores, quiet);

      util::PrintMsg(
          "--------------------------\n Elapsed: " + std::to_string(elapsed),
          !quiet);
    }

    //Add other switching parameters, if needed
    std::vector<std::string> switches{"advance-mode"};

    GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
                                    [
                                        ref_k_cores
    ](util::Parameters &parameters, GraphT &graph) {
                                      return app::kcore::RunTests(
                                          parameters, graph, ref_k_cores,
                                          util::DEVICE);
                                    }));

    if (!quick) {
      //Deallocate host references
      delete[] ref_k_cores;
      ref_k_cores = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test kcore");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::kcore::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  //Available graph types
  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B |
                           app::VALUET_F32B | app::UNDIRECTED>(
                           parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
