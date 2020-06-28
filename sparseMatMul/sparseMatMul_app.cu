// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file graphsum_app.cu
 *
 * @brief gcn graphsum application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/gcn/sparseMatMul/sparseMatMul_enactor.cuh>
#include <gunrock/app/gcn/sparseMatMul/sparseMatMul_test.cuh>

#include <gunrock/app/gcn/module.h>

/**
 * @brief      graphsum layer of GCN
 *
 * @param      parameters  The parameters
 * @param      graph       The graph
 * @param[in]  dim         dimension of the feature vector
 * @param      in          the input to the graphsum layer
 * @param      out         output matrix
 *
 * @tparam     GraphT      type of the graph
 * @tparam     ValueT      type of the value, double by default
 *
 * @return     time elapsed to execute
 */

namespace gunrock {
namespace app {
namespace sparseMatMul {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "inx", util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::REQUIRED_PARAMETER,
      "", "input file name to feature matrix", __FILE__, __LINE__
  ));
//  GUARD_CU(parameters.Use<std::string>(
//      "inw", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
//      "", "input file name to weight matrix", __FILE__, __LINE__
//  ));
  GUARD_CU(parameters.Use<int>(
      "hidden_dim", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "hidden dimension of weight matrix", __FILE__, __LINE__
  ));

//  GUARD_CU(parameters.Use<int>(
//      "dim", util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
//      "", "input file name to feature matrix", __FILE__, __LINE__
//  ));

return retval;
}


}
}
}

using namespace gunrock;

template <typename SizeT, typename ValueT, typename SpmatT>
struct sprmul : module {
  typedef app::sparseMatMul::Problem<SpmatT> ProblemT;
  typedef app::sparseMatMul::Enactor<ProblemT> EnactorT;

  SpmatT *a;
  util::Array1D<SizeT, ValueT> b, c, b_grad, c_grad;
  ProblemT *problem;
  EnactorT *enactor;
  int num_nodes, in_dim, out_dim;
  float *fw_time, *bw_time;

  sprmul(util::Parameters &p, SpmatT &_a, util::Array1D<SizeT, ValueT> &_b, util::Array1D<SizeT, ValueT> &_b_grad,
      util::Array1D<SizeT, ValueT> &_c, util::Array1D<SizeT, ValueT> &_c_grad, int _in_dim, int _out_dim,
      float *_fw_time, float *_bw_time) :
  a(&_a), b(_b), c(_c), b_grad(_b_grad), c_grad(_c_grad), in_dim(_in_dim), out_dim(_out_dim), fw_time(_fw_time),
  bw_time(_bw_time) {
    num_nodes = _a.nodes;
    problem = new ProblemT(p);
    enactor = new EnactorT();

    problem->Init(_a, in_dim, out_dim);
    enactor->Init(*problem);
  }

  virtual void forward(bool train) override {
    timer.Start ();

    problem->Reset(1, b, c);
    enactor->Reset();
    enactor->Enact();

    timer.Stop ();
    *fw_time += timer.ElapsedMillis ();
  }

  virtual void backward() override {
    timer.Start ();
    problem->Reset(0, c_grad, b_grad);
    enactor->Reset();
    enactor->Enact();

    timer.Stop ();
    *bw_time += timer.ElapsedMillis ();
  }
};

template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double sparseMatMul(gunrock::util::Parameters &parameters, GraphT &graph, const int dim,
                    const int out_dim, ValueT *in, ValueT *out) {
  typedef typename GraphT::VertexT VertexT;
  typedef gunrock::app::sparseMatMul::Problem<GraphT> ProblemT;
  typedef gunrock::app::sparseMatMul::Enactor<ProblemT> EnactorT;
  gunrock::util::CpuTimer cpu_timer;
  gunrock::util::Location target = gunrock::util::DEVICE;
  double total_time = 0;
  if (parameters.UseDefault("quiet")) parameters.Set("quiet", true);

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  problem.Init(graph, dim, out_dim, in, target);
  enactor.Init(problem, target);

  problem.Reset(in);
  enactor.Reset();

  cpu_timer.Start();
  enactor.Enact();
  cpu_timer.Stop();

  total_time += cpu_timer.ElapsedMillis();
  problem.Extract(out);

  enactor.Release(target);
  problem.Release(target);

  return total_time;
}

/*
 * @brief      Simple interface take in graph as CSR format
 *
 * @param[in]  num_nodes    Number of veritces in the input graph
 * @param[in]  num_edges    Number of edges in the input graph
 * @param[in]  row_offsets  CSR-formatted graph input row offsets
 * @param[in]  col_indices  CSR-formatted graph input column indices
 * @param[in]  dim          The dimenssion of the feature vector
 * @param      in           The input to graphsum layer
 * @param      out          The output of graphsum layer
 *
 * @tparam     VertexT      type of vertex id, default to int
 *
 * @return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, typename SizeT = int, typename ValueT = double>
double sparseMatMul(gunrock::util::Parameters &parameters, const SizeT n_rows, const SizeT nnz,
    SizeT *row_offsets, VertexT *col_indices, ValueT *vals,
    const int dim, const int outdim, ValueT *b, ValueT *c) {
  typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
                                           gunrock::graph::HAS_EDGE_VALUES |
                                               gunrock::graph::HAS_CSR>
      GraphT;
  typedef typename GraphT::CsrT CsrT;

  // Setup parameters
//  gunrock::util::Parameters parameters("sparseMatMul");
//  gunrock::graphio::UseParameters(parameters);
//  gunrock::app::sparseMatMul::UseParameters(parameters);
//  gunrock::app::UseParameters_test(parameters);
//  parameters.Parse_CommandLine(0, NULL);
//  parameters.Set("graph-type", "by-pass");
//
//  bool quiet = parameters.Get<bool>("quiet");
  GraphT graph;
  // Assign pointers into gunrock graph format
  graph.CsrT::Allocate(n_rows, nnz, gunrock::util::HOST);
  graph.CsrT::row_offsets.SetPointer(row_offsets, n_rows + 1, gunrock::util::HOST);
  graph.CsrT::column_indices.SetPointer(col_indices, nnz, gunrock::util::HOST);
  graph.CsrT::edge_values.SetPointer(vals, nnz, gunrock::util::HOST);

//  graph.CsrT::row_offsets.Print();
//  graph.CsrT::column_indices.Print();
//  graph.CsrT::edge_values.Print();

//  graph.Display();
  graph.nodes = n_rows;
  graph.edges = nnz;
  gunrock::graphio::LoadGraph(parameters, graph);

  // Run the gcn_graphsum
  double elapsed_time = sparseMatMul(parameters, graph, dim, outdim, b, c);

  // Cleanup
  graph.Release();

  return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
