// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_rw.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <iostream>
#include <string>
#include <fstream>

#include <gunrock/app/rw/rw_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::rw;

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

    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_NODE_VALUES | graph::HAS_CSR>
        GraphT;
    typedef typename GraphT::CsrT CsrT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    int walk_length = parameters.Get<int>("walk-length");
    int walks_per_node = parameters.Get<int>("walks-per-node");
    int walk_mode = parameters.Get<int>("walk-mode");
    bool store_walks = parameters.Get<bool>("store-walks");
    VertexT *ref_walks = NULL;

    ValueT *node_values = NULL;
    if (walk_mode != 0) {
      std::string node_value_path =
          parameters.Get<std::string>("node-value-path");
      if (node_value_path.compare("") == 0) {
        printf("test_rw: `node-value-path` must be set if `walk-mode` != 0");
        return retval;
      }

      std::ifstream node_value_file(node_value_path, std::ios_base::in);
      for (int i = 0; i < graph.nodes; i++) {
        node_value_file >> graph.node_values[i];
      }
    }

    if (!quick) {
      if (store_walks) {
        ref_walks = new VertexT[graph.nodes * walk_length * walks_per_node];
      }

      util::PrintMsg("__________________________", !quiet);

      float elapsed = APP_NAMESPACE::CPU_Reference(
          graph.csr(), walk_length, walks_per_node, walk_mode, store_walks,
          ref_walks, quiet);

      util::PrintMsg(
          "--------------------------\n Elapsed: " + std::to_string(elapsed),
          !quiet);
    }

    std::vector<std::string> switches{"advance-mode"};
    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [walk_length, walks_per_node, walk_mode, ref_walks, store_walks](
            util::Parameters &parameters, GraphT &graph) {
          return APP_NAMESPACE::RunTests(parameters, graph, walk_length,
                                         walks_per_node, walk_mode, store_walks,
                                         ref_walks, util::DEVICE);
        }));

    if (!quick) {
      delete[] ref_walks;
      ref_walks = NULL;
    }
    if (walk_mode != 0) {
      delete[] node_values;
      node_values = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test rw");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(APP_NAMESPACE::UseParameters(parameters));
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
