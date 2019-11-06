/// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sage.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

#include <gunrock/app/sage/sage_app.cu>
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
                                    // graph::HAS_EDGE_VALUES | graph::HAS_CSR>
                                    graph::HAS_CSR>
        GraphT;
    typedef typename GraphT::CsrT CsrT;

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

    std::vector<std::string> switches{
        "feature-column", "num-children-per-source", "num-leafs-per-child"};
    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [](util::Parameters &parameters, GraphT &graph) {
          cudaError_t retval = cudaSuccess;

          bool quick = parameters.Get<bool>("quick");
          if (!quick) {
            bool quiet = parameters.Get<bool>("quiet");
            std::string wf1_file = parameters.Get<std::string>("Wf1");
            std::string wa1_file = parameters.Get<std::string>("Wa1");
            std::string wf2_file = parameters.Get<std::string>("Wf2");
            std::string wa2_file = parameters.Get<std::string>("Wf2");
            std::string feature_file = parameters.Get<std::string>("features");
            int Wf1_dim_0 =
                parameters.Get<int>("feature-column");  //("Wf1-dim0");
            int Wa1_dim_0 =
                parameters.Get<int>("feature-column");  //("Wa1-dim0");
            int Wf1_dim_1 = parameters.Get<int>("Wf1-dim1");
            int Wa1_dim_1 = parameters.Get<int>("Wa1-dim1");
            int Wf2_dim_0 =
                Wf1_dim_1 + Wa1_dim_1;  // parameters.Get<int> ("Wf2-dim0");
            int Wa2_dim_0 =
                Wf1_dim_1 + Wa1_dim_1;  // parameters.Get<int> ("Wa2-dim0");
            int Wf2_dim_1 = parameters.Get<int>("Wf2-dim1");
            int Wa2_dim_1 = parameters.Get<int>("Wa2-dim1");
            int num_neigh1 = parameters.Get<int>("num-children-per-source");
            int num_neigh2 = parameters.Get<int>("num-leafs-per-child");
            if (!util::isValid(num_neigh2)) num_neigh2 = num_neigh1;
            int batch_size = parameters.Get<int>("batch-size");

            ValueT **W_f_1 = app::sage::template ReadMatrix<ValueT, SizeT>(
                wf1_file, Wf1_dim_0, Wf1_dim_1);
            ValueT **W_a_1 = app::sage::template ReadMatrix<ValueT, SizeT>(
                wa1_file, Wa1_dim_0, Wa1_dim_1);
            ValueT **W_f_2 = app::sage::template ReadMatrix<ValueT, SizeT>(
                wf2_file, Wf2_dim_0, Wf2_dim_1);
            ValueT **W_a_2 = app::sage::template ReadMatrix<ValueT, SizeT>(
                wa2_file, Wa2_dim_0, Wa2_dim_1);
            ValueT **features = app::sage::template ReadMatrix<ValueT, SizeT>(
                feature_file, graph.nodes, Wf1_dim_0);
            ValueT *source_embedding =
                new ValueT[(uint64_t)graph.nodes * (Wa2_dim_1 + Wf2_dim_1)];
            util::PrintMsg("Computing reference value ...", !quiet);
            util::PrintMsg("__________________________", !quiet);
            float elapsed = app::sage::CPU_Reference(
                parameters, graph, features, W_f_1, W_a_1, W_f_2, W_a_2,
                source_embedding, quiet);
            util::PrintMsg(
                "--------------------------\n"
                "CPU Reference elapsed: " +
                    std::to_string(elapsed) + " ms.",
                !quiet);
            app::sage::Validate_Results(parameters, graph, source_embedding,
                                        Wa2_dim_1 + Wf2_dim_1, true);
            delete[] source_embedding;
            source_embedding = NULL;
            for (auto v = 0; v < graph.nodes; v++) {
              delete[] features[v];
              features[v] = NULL;
            }
            delete[] features;
            features = NULL;
          }

          std::vector<std::string> switches2{"batch-size"};
          GUARD_CU(app::Switch_Parameters(
              parameters, graph, switches2,
              [](util::Parameters &parameters, GraphT &graph) {
                return app::sage::RunTests(parameters, graph);
              }));
          return retval;
        }));

    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test sage");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::sage::UseParameters(parameters));
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
