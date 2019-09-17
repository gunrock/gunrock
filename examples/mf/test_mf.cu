// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_mf.cu
 *
 * @brief Simple test driver program for max-flow algorithm.
 */

#include <gunrock/app/mf/mf_app.cu>
#include <gunrock/app/mf/mf_helpers.cuh>
#include <gunrock/app/test_base.cuh>

#define debug_aml(a...)
//#define debug_aml(a...) {printf(a); printf("\n");}

using namespace gunrock;

/*****************************************************************************
 * Main
 *****************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT	  Type of vertex identifier
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
    typedef typename GraphT::CsrT CsrT;
    cudaError_t retval = cudaSuccess;
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    //
    // Load Graph
    //
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    bool undirected;
    parameters.Get("undirected", undirected);

    GraphT d_graph;
    if (not undirected) {
      debug_aml("Load directed graph");
      parameters.Set<int>("remove-duplicate-edges", false);
      GUARD_CU(graphio::LoadGraph(parameters, d_graph));
    }

    debug_aml("Load undirected graph");
    GraphT u_graph;
    parameters.Set<int>("undirected", 1);
    parameters.Set<int>("remove-duplicate-edges", true);
    GUARD_CU(graphio::LoadGraph(parameters, u_graph));

    cpu_timer.Stop();

    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    VertexT source = parameters.Get<VertexT>("source");
    VertexT sink = parameters.Get<VertexT>("sink");
    int num_repeats = parameters.Get<int>("num-repeats");

    if (num_repeats == util::PreDefinedValues<int>::InvalidValue) {
      num_repeats =
          max(10, static_cast<int>(pow(10, floor(log10(u_graph.nodes)))));
      parameters.Set<int>("num-repeats", num_repeats);
    }

    util::PrintMsg("Number of ForAll() repeats per iteration: " +
                       std::to_string(num_repeats),
                   !quiet);

    if (source == util::PreDefinedValues<VertexT>::InvalidValue ||
        source >= u_graph.nodes) {
      source = u_graph.nodes - 2;
      parameters.Set("source", source);
    }
    if (sink == util::PreDefinedValues<VertexT>::InvalidValue ||
        sink >= u_graph.nodes) {
      sink = u_graph.nodes - 1;
      parameters.Set("sink", sink);
    }

    if (not undirected) {
      debug_aml("Directed graph:");
      debug_aml("number of edges %d", d_graph.edges);
      debug_aml("number of nodes %d", d_graph.nodes);
    }

    debug_aml("Undirected graph:");
    debug_aml("number of edges %d", u_graph.edges);
    debug_aml("number of nodes %d", u_graph.nodes);

    std::map<std::pair<VertexT, VertexT>, SizeT> d_edge_id;
    std::map<std::pair<VertexT, VertexT>, SizeT> u_edge_id;
    app::mf::GetEdgesIds(d_graph, d_edge_id);
    app::mf::GetEdgesIds(u_graph, u_edge_id);

    ValueT *flow_edge = NULL;

    util::Array1D<SizeT, VertexT> reverse;
    GUARD_CU(reverse.Allocate(u_graph.edges, util::HOST));
    app::mf::InitReverse(u_graph, u_edge_id, reverse.GetPointer(util::HOST));

    if (not undirected) {
      // Correct capacity values on reverse edges
      app::mf::CorrectCapacity(u_graph, d_graph, d_edge_id);
    }

    //
    // Compute reference CPU max flow algorithm.
    //
    ValueT max_flow = util::PreDefinedValues<ValueT>::InvalidValue;

    if (!quick) {
      util::PrintMsg("______CPU reference algorithm______", true);
      flow_edge = (ValueT *)malloc(sizeof(ValueT) * u_graph.edges);
      double elapsed = app::mf::CPU_Reference(
          parameters, u_graph, u_edge_id, source, sink, max_flow,
          reverse.GetPointer(util::HOST), flow_edge);
      util::PrintMsg("-----------------------------------\nElapsed: " +
                         std::to_string(elapsed) +
                         " ms\n Max flow CPU = " + std::to_string(max_flow),
                     true);
    }

    std::vector<std::string> switches{"advance-mode"};
    GUARD_CU(app::Switch_Parameters(
        parameters, u_graph, switches,
        [flow_edge, reverse, max_flow](util::Parameters &parameters,
                                       GraphT &u_graph) {
          return app::mf::RunTests(parameters, u_graph,
                                   reverse.GetPointer(util::HOST), flow_edge,
                                   max_flow);
        }));

    // Clean up
    free(flow_edge);
    GUARD_CU(reverse.Release());

    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test mf");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::mf::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::SIZET_U32B |
                           app::VALUET_F64B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
