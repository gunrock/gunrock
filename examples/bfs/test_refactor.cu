#include <stdio.h>
#include <iostream>
//#include <gunrock/util/array_utils.cuh>
//#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
//#include <gunrock/oprtr/1D_oprtr/for_each.cuh>
//#include <gunrock/oprtr/1D_oprtr/1D_scalar.cuh>
//#include <gunrock/oprtr/1D_oprtr/1D_1D.cuh>
//#include <gunrock/graph/csr.cuh>
//#include <gunrock/graph/coo.cuh>
//#include <gunrock/graph/csc.cuh>
//#include <gunrock/graph/gp.cuh>
#include <gunrock/graphio/graphio.cuh>

//#include <gunrock/app/frontier.cuh>

//#include <gunrock/partitioner/partitioner.cuh>

//#include <gunrock/app/sssp/sssp_problem.cuh>

//#include <gunrock/app/enactor_base.cuh>

#include <gunrock/app/sssp/sssp_enactor.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::graph;
using namespace gunrock::app;

typedef uint32_t VertexT;
typedef unsigned long long SizeT;
typedef float ValueT;

template <typename _VertexT = int, typename _SizeT = _VertexT,
          typename _ValueT = _VertexT, GraphFlag _FLAG = GRAPH_NONE,
          unsigned int _cudaHostRegisterFlag = cudaHostRegisterDefault>
struct TestGraph : public Csr<VertexT, SizeT, ValueT,
                              _FLAG | HAS_CSR | HAS_COO | HAS_CSC | HAS_GP,
                              _cudaHostRegisterFlag>,
                   public Coo<VertexT, SizeT, ValueT,
                              _FLAG | HAS_CSR | HAS_COO | HAS_CSC | HAS_GP,
                              _cudaHostRegisterFlag>,
                   public Csc<VertexT, SizeT, ValueT,
                              _FLAG | HAS_CSR | HAS_COO | HAS_CSC | HAS_GP,
                              _cudaHostRegisterFlag>,
                   public Gp<VertexT, SizeT, ValueT,
                             _FLAG | HAS_CSR | HAS_COO | HAS_CSC | HAS_GP,
                             _cudaHostRegisterFlag> {
  typedef _VertexT VertexT;
  typedef _SizeT SizeT;
  typedef _ValueT ValueT;
  static const GraphFlag FLAG = _FLAG | HAS_CSR | HAS_COO | HAS_CSC | HAS_GP;
  static const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;
  typedef Csr<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CsrT;
  typedef Coo<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CooT;
  typedef Csc<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CscT;
  typedef Gp<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> GpT;

  SizeT nodes, edges;

  template <typename CooT_in>
  cudaError_t FromCoo(CooT_in &coo, bool self_coo = false) {
    cudaError_t retval = cudaSuccess;
    nodes = coo.CooT::nodes;
    edges = coo.CooT::edges;
    retval = this->CsrT::FromCoo(coo);
    if (retval) return retval;
    retval = this->CscT::FromCoo(coo);
    if (retval) return retval;
    if (!self_coo) retval = this->CooT::FromCoo(coo);
    return retval;
  }

  template <typename CsrT_in>
  cudaError_t FromCsr(CsrT_in &csr, bool self_csr = false) {
    cudaError_t retval = cudaSuccess;
    nodes = csr.CsrT::nodes;
    edges = csr.CsrT::edges;
    retval = this->CooT::FromCsr(csr);
    if (retval) return retval;
    retval = this->CscT::FromCsr(csr);
    if (retval) return retval;
    if (!self_csr) retval = this->CsrT::FromCsr(csr);
    return retval;
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;

    util::PrintMsg("GraphT::Realeasing on " + util::Location_to_string(target));
    retval = this->CooT::Release(target);
    if (retval) return retval;
    retval = this->CsrT::Release(target);
    if (retval) return retval;
    retval = this->CscT::Release(target);
    if (retval) return retval;
    retval = this->GpT::Release(target);
    if (retval) return retval;
    return retval;
  }

  CsrT &csr() { return (static_cast<CsrT *>(this))[0]; }
};

/*void Test_Array()
{
    // test array
    Array1D<int, int, PINNED> test_array;
    test_array.SetName("test_array");
    test_array.Allocate(1024, HOST | DEVICE);
    test_array.EnsureSize(2048);
    test_array.Move(HOST, DEVICE);
    test_array.Release();
}

void Test_ForAll()
{
    // test ForAll
    Array1D<int, int, PINNED> array1, array2;
    array1.SetName("array1"); array2.SetName("array2");
    array1.Allocate(1024 * 1024, HOST | DEVICE);
    array2.Allocate(1024 * 1024, HOST | DEVICE);

    array1.ForAll(
        [] __host__ __device__ (int* elements, int pos)
        {
            elements[pos] = pos / 1024;
        });//, DefaultSize, HOST | DEVICE);
    array2.ForAll(
        [] __host__ __device__ (int* elements, int pos){
            elements[pos] = pos % 1024;
        });//, DefaultSize, HOST | DEVICE);
    //ForAll(array1, 1024 * 1024,
    //    [] __host__ __device__ (int* elements, int pos){
    //        printf("array1[%d] = %d\t", pos, elements[pos]);
    //    }, HOST | DEVICE);
    int mod = 10;
    std::cout << "mod = ?";
    std::cin >> mod;
    array1.ForAllCond( array2,
        [mod] __host__ __device__ (int* elements_in, int* elements_out, int pos)
        {
            return (elements_in[pos] == elements_out[pos] && (pos%mod) == 0);
        },
        [mod] __host__ __device__ (int* elements_in, int* elements_out, int pos)
        {
            //if (elements_in[pos] == elements_out[pos] && (pos%mod) == 0)
                printf("on %s: array1[%d] = array2[%d] = %d\n",
#ifdef __CUDA_ARCH__
                    "GPU",
#else
                    "CPU",
#endif
                    pos, pos, elements_in[pos]);
        });//, DefaultSize, HOST | DEVICE);
    cudaDeviceSynchronize();
}

void Test_ForEach()
{
    // test ForEach
    Array1D<SizeT, ValueT, PINNED> array3, array4;
    array3.SetName("array3");array4.SetName("array4");
    SizeT length = 1024 * 1024;
    Location target = HOST | DEVICE;
    array3.Allocate(length, target);
    array4.Allocate(length, target);
    array4.SetIdx();
    array3 = 10;
    array3 += array4;
    array3 -= 19.5;
    //ForEach(array3.GetPointer(DEVICE),
    //    [] __host__ __device__ (ValueT &element){
    //        element = 10;
    //    }, length, DEVICE);
    array4.ForEach([] __host__ __device__ (ValueT &element){
            element = 20;
        });
}

void Test_Csr()
{
    // Test_Csr
    typedef int VertexT;
    Csr<VertexT, SizeT, ValueT, HAS_CSR> csr;
    csr.Allocate(10, 10);
    Coo<VertexT, SizeT, ValueT, HAS_COO> coo;
    csr.FromCoo(coo);

    Csr<VertexT, SizeT, ValueT, HAS_CSR | HAS_EDGE_VALUES> csr2;
    csr2.Allocate(10, 10);
    Coo<VertexT, SizeT, ValueT, HAS_COO | HAS_EDGE_VALUES> coo2;
    csr2.FromCoo(coo2);
}

cudaError_t Test_GraphIo(int argc, char* argv[])
{
    // Test graphio
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test refactor");
    typedef TestGraph<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> GraphT;
    GraphT graph;

    retval = graphio::UseParameters(parameters);
    if (retval) return retval;
    retval = parameters.Parse_CommandLine(argc, argv);
    if (retval) return retval;
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }

    retval = parameters.Check_Required();
    if (retval) return retval;
    retval = graphio::LoadGraph(parameters, graph);
    if (retval) return retval;
    //retval = graph.CooT::Display();
    if (retval) return retval;
    //retval = graph.CsrT::Display();
    if (retval) return retval;
    //retval = graph.CscT::Display();
    if (retval) return retval;

    typedef Csr<VertexT, SizeT, ValueT> CsrT;
    CsrT csr;
    retval = csr.FromCoo(graph);
    if (retval) return retval;
    PrintMsg("CSR from COO:");
    csr.Display();
    //graph.CooT::Display();

    retval = csr.FromCsc(graph);
    if (retval) return retval;
    PrintMsg("CSR from CSC:");
    csr.Display();
    //graph.CooT::Display();

    retval = csr.FromCsr(graph);
    if (retval) return retval;
    PrintMsg("CSR from CSR:");
    csr.Display();
    //graph.CooT::Display();

    typedef Csr<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> CsreT;
    CsreT csre;
    retval = csre.FromCoo(graph);
    if (retval) return retval;
    PrintMsg("CSRE from COO:");
    csre.Display();
    //graph.CooT::Display();

    retval = csre.FromCsc(graph);
    if (retval) return retval;
    PrintMsg("CSRE from CSC:");
    csre.Display();
    //graph.CooT::Display();

    retval = csre.FromCsr(graph);
    if (retval) return retval;
    PrintMsg("CSRE from CSR:");
    csre.Display();
    //graph.CooT::Display();

    typedef Csc<VertexT, SizeT, ValueT> CscT;
    CscT csc;
    retval = csc.FromCoo(graph);
    if (retval) return retval;
    PrintMsg("CSC from COO:");
    csc.Display();
    //graph.CooT::Display();

    retval = csc.FromCsc(graph);
    if (retval) return retval;
    PrintMsg("CSC from CSC:");
    csc.Display();
    //graph.CooT::Display();

    retval = csc.FromCsr(graph);
    if (retval) return retval;
    PrintMsg("CSC from CSR:");
    csc.Display();
    //graph.CooT::Display();

    typedef Csc<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> CsceT;
    CsceT csce;
    retval = csce.FromCoo(graph);
    if (retval) return retval;
    PrintMsg("CSCE from COO:");
    csce.Display();
    //graph.CooT::Display();

    retval = csce.FromCsc(graph);
    if (retval) return retval;
    PrintMsg("CSCE from CSC:");
    csce.Display();
    //graph.CooT::Display();

    retval = csce.FromCsr(graph);
    if (retval) return retval;
    PrintMsg("CSCE from CSR:");
    csce.Display();
    //graph.CooT::Display();

    typedef Coo<VertexT, SizeT, ValueT> CooT;
    CooT coo;
    retval = coo.FromCoo(graph);
    if (retval) return retval;
    PrintMsg("COO from COO:");
    coo.Display();
    //graph.CooT::Display();

    retval = coo.FromCsc(graph);
    if (retval) return retval;
    PrintMsg("COO from CSC:");
    coo.Display();
    //graph.CooT::Display();

    retval = coo.FromCsr(graph);
    if (retval) return retval;
    PrintMsg("COO from CSR:");
    coo.Display();
    //graph.CooT::Display();

    typedef Coo<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> CooeT;
    CooeT cooe;
    retval = cooe.FromCoo(graph);
    if (retval) return retval;
    PrintMsg("COOE from COO:");
    cooe.Display();
    //graph.CooT::Display();

    retval = cooe.FromCsc(graph);
    if (retval) return retval;
    PrintMsg("COOE from CSC:");
    cooe.Display();
    //graph.CooT::Display();

    retval = cooe.FromCsr(graph);
    if (retval) return retval;
    PrintMsg("COOE from CSR:");
    cooe.Display();
    //graph.CooT::Display();

    return retval;
}

void Test_Frontier()
{
    Frontier<VertexT, SizeT> frontier;
    frontier.SetName("test_frontier");
    frontier.Init();
    frontier.Release();
}*/

template <typename GraphT>
cudaError_t LoadGraph(util::Parameters &parameters, GraphT &graph) {
  cudaError_t retval = cudaSuccess;

  retval = graphio::LoadGraph(parameters, graph);
  if (retval) return retval;
  // util::cpu_mt::PrintCPUArray<typename GraphT::SizeT, typename
  // GraphT::SizeT>(
  //    "row_offsets", graph.GraphT::CsrT::row_offsets + 0, graph.nodes+1);

  // util::cpu_mt::PrintCPUArray<typename GraphT::SizeT, typename
  // GraphT::ValueT>(
  //    "edge_values", graph.GraphT::CsrT::edge_values + 0, graph.edges);
  return retval;
}

/*template <typename GraphT>
cudaError_t Test_Partitioner(Parameters &parameters, GraphT &graph)
{
    cudaError_t retval = cudaSuccess;
    GraphT *sub_graphs = NULL;

    retval = partitioner::Partition(
        graph, sub_graphs, parameters, 4,
        partitioner::Keep_Node_Num, util::HOST);
    if (retval) return retval;

    PrintMsg("OrgGraph");
    graph.CsrT::Display();

    for (int i=0; i<4; i++)
    {
        PrintMsg("SubGraph " + std::to_string(i));
        sub_graphs[i].CsrT::Display();
    }
    return retval;
}*/

/*template <typename GraphT>
cudaError_t Test_ProblemBase(Parameters &parameters, GraphT &graph)
{
    cudaError_t retval = cudaSuccess;

    typedef ProblemBase<GraphT> ProblemT;
    ProblemT problem;

    retval = problem.Init(parameters, graph);
    if (retval) return retval;

    return retval;
}*/

template <typename GraphT>
cudaError_t Test_SSSP(Parameters &parameters, GraphT &graph,
                      util::Location target = util::HOST) {
  cudaError_t retval = cudaSuccess;

  typedef gunrock::app::sssp::Problem<GraphT, unsigned char> ProblemT;
  typedef gunrock::app::sssp::Enactor<ProblemT> EnactorT;
  ProblemT problem;
  EnactorT enactor;

  retval = problem.Init(parameters, graph, target);
  if (retval) return retval;
  retval = enactor.Init(parameters, &problem, target);
  if (retval) return retval;

  retval = problem.Reset(0, target);
  if (retval) return retval;
  retval = enactor.Reset(0, target);
  if (retval) return retval;

  retval = enactor.Enact(0);
  if (retval) return retval;

  retval = problem.Release(target);
  if (retval) return retval;
  retval = enactor.Release(target);
  if (retval) return retval;
  return retval;
}

/*template <typename GraphT>
cudaError_t Test_EnactorBase(Parameters &parameters, GraphT &graph,
util::Location target = util::HOST)
{
    cudaError_t retval = cudaSuccess;

    typedef gunrock::app::EnactorBase<GraphT> EnactorT;
    EnactorT enactor("refactor");

    retval = enactor.Init(parameters,  2, NULL, 1024, target);
    if (retval) return retval;

    retval = enactor.Reset(target);
    if (retval) return retval;

    retval = enactor.Release(target);
    if (retval) return retval;

    return retval;
}*/

int main(int argc, char *argv[]) {
  // const SizeT DefaultSize = PreDefinedValues<SizeT>::InvalidValue;
  // Test_Array();
  // Test_ForAll();
  // Test_ForEach();
  // Test_Csr();
  // Test_GraphIo(argc, argv);
  // Test_Frontier();

  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test refactor");
  typedef TestGraph<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> GraphT;
  GraphT graph;

  GUARD_CU(graphio::UseParameters(parameters));
  // GUARD_CU(partitioner::UseParameters(parameters));
  GUARD_CU(app::sssp::UseParameters(parameters));
  GUARD_CU(app::sssp::UseParameters2(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }

  retval = parameters.Check_Required();
  if (retval) return 5;

  retval = LoadGraph(parameters, graph);
  if (retval) return 11;
  // retval = Test_Partitioner(parameters, graph);
  // if (retval) return 12;
  /*retval = Test_SSSPProblem(parameters, graph, util::HOST);
  if (retval) return 13;
  util::PrintMsg("====Test on HOST finished");

  retval = Test_SSSPProblem(parameters, graph, util::DEVICE);
  if (retval) return 14;
  util::PrintMsg("====Test on DEVICE finished");

  retval = Test_SSSPProblem(parameters, graph, util::HOST | util::DEVICE);
  if (retval) return 15;
  util::PrintMsg("====Test on HOST | DEVICE finished");*/

  /*retval = Test_EnactorBase(parameters, graph, util::HOST);
  if (retval) return 16;
  util::PrintMsg("====Test on HOST finished");

  retval = Test_EnactorBase(parameters, graph, util::DEVICE);
  if (retval) return 17;
  util::PrintMsg("====Test on DEVICE finished");

  retval = Test_EnactorBase(parameters, graph, util::HOST | util::DEVICE);
  if (retval) return 18;
  util::PrintMsg("====Test on HOST | DEVICE finished");*/

  // retval = Test_SSSP(parameters, graph, util::HOST);
  // if (retval) return 16;
  // util::PrintMsg("====Test on HOST finished");

  retval = Test_SSSP(parameters, graph, util::DEVICE);
  if (retval) return 17;
  util::PrintMsg("====Test on DEVICE finished");

  // retval = Test_SSSP(parameters, graph, util::HOST | util::DEVICE);
  // if (retval) return 18;
  // util::PrintMsg("====Test on HOST | DEVICE finished");
  return 0;
}
