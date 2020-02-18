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
                                    graph::HAS_EDGE_VALUES | graph::HAS_CSR | graph::HAS_DYN>
        GraphT;
    typedef typename GraphT::CsrT CsrT;
    typedef typename GraphT::DynT DynT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;  // graph we process on

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    // force edge values to be 1, don't enable this unless you really want to
    // for (SizeT e=0; e < graph.edges; e++)
    //    graph.CsrT::edge_values[e] = 1;
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());
    // GUARD_CU(graph.CsrT::edge_values.Print("", 100));
    // util::PrintMsg("sizeof(VertexT) = " + std::to_string(sizeof(VertexT))
    //    + ", sizeof(SizeT) = " + std::to_string(sizeof(SizeT))
    //    + ", sizeof(ValueT) = " + std::to_string(sizeof(ValueT)));


    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test sssp");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B |
                           app::SIZET_U32B | 
                           app::VALUET_S32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());


  //return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
  //                         app::SIZET_U32B | app::SIZET_U64B |
  //                         app::VALUET_S32B | app::DIRECTED | app::UNDIRECTED>(
  //    parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
