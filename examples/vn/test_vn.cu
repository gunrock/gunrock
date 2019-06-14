// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_vn.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

#include <gunrock/app/vn/vn_app.cu>
#include <gunrock/app/test_base.cuh>
#include <assert.h>

using namespace gunrock;

/******************************************************************************
 * Main
 ******************************************************************************/

template <typename GraphT, typename ValueT = typename GraphT::ValueT,
          typename VertexT = typename GraphT::VertexT>
cudaError_t vn_set_srcs(util::Parameters &parameters, GraphT &graph) {
  /*
      Helper for randomly seeding VN
      This is slightly different from the standard `SetSrcs` function
      because we have to set multiple batches of multiple seeds
  */
  cudaError_t retval = cudaSuccess;
  std::string src = parameters.Get<std::string>("src");
  std::vector<VertexT> srcs;
  if (src == "random") {
    int src_seed = parameters.Get<int>("src-seed");
    int num_runs = parameters.Get<int>("num-runs");
    int srcs_per_run = parameters.Get<int>("srcs-per-run");
    if (!util::isValid(src_seed)) {
      src_seed = time(NULL);
      GUARD_CU(parameters.Set<int>("src-seed", src_seed));
    }
    srand(src_seed);

    std::vector<VertexT> run_srcs;
    for (int i = 0; i < num_runs; i++) {
      for (int j = 0; j < srcs_per_run; j++) {
        bool src_valid = false;
        VertexT v;
        while (!src_valid) {
          v = rand() % graph.nodes;
          if (std::find(run_srcs.begin(), run_srcs.end(), v) ==
              run_srcs.end()) {
            if (graph.GetNeighborListLength(v) != 0) {
              src_valid = true;
            }
          }
        }
        srcs.push_back(v);
        run_srcs.push_back(v);
      }
      run_srcs.clear();
    }
    GUARD_CU(parameters.Set<std::vector<VertexT>>("srcs", srcs));
  } else {
    GUARD_CU(parameters.Set("srcs", src));
  }

  return retval;
}

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
    typedef typename GraphT::CsrT CsrT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;  // graph we process on

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    // Load parameters
    vn_set_srcs(parameters, graph);

    std::vector<VertexT> srcs_vector =
        parameters.Get<std::vector<VertexT>>("srcs");
    int total_num_srcs = srcs_vector.size();
    int num_runs = parameters.Get<int>("num-runs");
    int srcs_per_run = parameters.Get<int>("srcs-per-run");
    if (srcs_per_run == util::PreDefinedValues<int>::InvalidValue) {
      srcs_per_run = total_num_srcs;
    }
    assert(total_num_srcs == num_runs * srcs_per_run);
    VertexT *all_srcs = &srcs_vector[0];

    ValueT **ref_distances = NULL;
    bool quick = parameters.Get<bool>("quick");
    if (!quick) {
      SizeT nodes = graph.nodes;
      bool quiet = parameters.Get<bool>("quiet");
      ref_distances = new ValueT *[num_runs];

      for (int run_num = 0; run_num < num_runs; run_num++) {
        VertexT *srcs = new VertexT[srcs_per_run];
        for (SizeT i = 0; i < srcs_per_run; i++) {
          srcs[i] = all_srcs[run_num * srcs_per_run + i % total_num_srcs];
        }

        ref_distances[run_num] = new ValueT[nodes];

        util::PrintMsg("__________________________", !quiet);
        float elapsed =
            app::vn::CPU_Reference(graph.csr(), ref_distances[run_num], NULL,
                                   srcs, srcs_per_run, quiet, false);

        std::string src_msg = "";
        for (SizeT i = 0; i < srcs_per_run; ++i) {
          src_msg += std::to_string(srcs[i]);
          if (i != srcs_per_run - 1) src_msg += ",";
        }
        util::PrintMsg("--------------------------\nRun " +
                           std::to_string(run_num) + " elapsed: " +
                           std::to_string(elapsed) + " ms, srcs = " + src_msg,
                       !quiet);
      }
    }

    std::vector<std::string> switches{"mark-pred", "advance-mode"};
    GUARD_CU(app::Switch_Parameters(
        parameters, graph, switches,
        [ref_distances](util::Parameters &parameters, GraphT &graph) {
          return app::vn::RunTests(parameters, graph, ref_distances);
        }));

    if (!quick) {
      for (int run_num = 0; run_num < num_runs; run_num++) {
        delete[] ref_distances[run_num];
        ref_distances[run_num] = NULL;
      }
      delete[] ref_distances;
      ref_distances = NULL;
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test vn");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::vn::UseParameters(parameters));
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
