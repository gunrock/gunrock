// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * test_mst.cu
 *
 * @brief Simple test driver for computing Minimum Spanning Tree.
 */

#include <stdio.h>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utilities
#include <gunrock/graphio/market.cuh>

// MST includes
#include <gunrock/app/cc/cc_app.cu>
#include <gunrock/app/mst/mst_enactor.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <moderngpu.cuh>

// CPU Kruskal MST reference
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::mst;

///////////////////////////////////////////////////////////////////////////////
// Housekeeping and utility routines
///////////////////////////////////////////////////////////////////////////////
void Usage()
{
  printf(
    " ------------------------------------------------------------------\n"
    " test_mst <graph type> <graph type args> [--device=<device_index>]\n"
    " [--instrumented] [--quick] [--v]\n\n"
    "Graph types and args:\n"
    "  market [<file>]\n"
    "    Reads a Matrix-Market coordinate-format graph of directed/undirected\n"
    "    edges from STDIN (or from the optionally-specified file)\n"
    "  --device=<device_index> Set GPU device for running the graph primitive\n"
    "  --instrumented If set then kernels keep track of queue-search_depth\n"
    "      and barrier duty (a relative indicator of load imbalance)\n"
    "  --quick If set will skip the CPU validation code\n"
    "  --v If set will enable debug mode\n\n"
    " ------------------------------------------------------------------\n");
}

/**
 * @brief Test_Parameter structure.
 */
struct MST_Test_Parameter : gunrock::app::TestParameter_Base
{
 public:
  MST_Test_Parameter()
  {

  }

  ~MST_Test_Parameter()
  {

  }

  void Init(CommandLineArgs &args)
  {
    TestParameter_Base::Init(args);
  }
};

/**
 * @brief Displays the MST result.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph.
 * @param[in] mst_output Pointer to the MST edge mask.
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(
  const Csr<VertexId, Value, SizeT> &graph, int *mst_output)
{
  fflush(stdout);
  int count = 0;
  int print_limit = graph.nodes;
  if (print_limit > 10)
  {
      print_limit = 10;
  }

  // find source vertex ids for display results
  VertexId *source = new VertexId[graph.edges];
  for (int i = 0; i < graph.nodes; ++i)
  {
    for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
    {
      source[j] = i;
    }
  }

  // print source-destination pairs of minimum spanning tree edges
  printf("GPU Minimum Spanning Tree [First %d edges]\n", print_limit);
  printf("src dst\n");
  for (int i = 0; i < graph.edges; ++i)
  {
    if (mst_output[i] == 1 && count <= print_limit)
    {
      printf("%d %d\n", source[i], graph.column_indices[i]);
      ++count;
    }
  }

  // clean up if necessary
  if (source) { delete [] source; }
}

///////////////////////////////////////////////////////////////////////////////
// CPU validation routines
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief A simple CPU-based reference MST implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] edge_values Weight value associated with each edge.
 * @param[in] graph Reference to the CSR graph we process on.
 *
 *  \return long long int which indicates the total weight of the graph.
 */
template<typename VertexId, typename Value, typename SizeT>
Value SimpleReferenceMST(
  const Value *edge_values, const Csr<VertexId, Value, SizeT> &graph)
{
  printf("\nMST CPU REFERENCE TEST\n");

  // Kruskal's minimum spanning tree preparations
  using namespace boost;
  typedef adjacency_list< vecS, vecS, undirectedS,
    no_property, property<edge_weight_t, int> > Graph;
  typedef graph_traits < Graph >::edge_descriptor   Edge;
  typedef graph_traits < Graph >::vertex_descriptor Vertex;
  typedef std::pair<VertexId, VertexId> E;

  E *edge_pairs = new E[graph.edges];
  int idx = 0;
  for (int i = 0; i < graph.nodes; ++i)
  {
    for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
    {
      edge_pairs[idx++] = std::make_pair(i, graph.column_indices[j]);
    }
  }

  Graph g(edge_pairs, edge_pairs + graph.edges, edge_values, graph.nodes);
  property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
  std::vector < Edge > spanning_tree;

  CpuTimer cpu_timer; // record the kernel running time
  cpu_timer.Start();

  // compute reference using kruskal_min_spanning_tree algorithm
  kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

  cpu_timer.Stop();
  float elapsed_cpu = cpu_timer.ElapsedMillis();

  // analyze reference results
  SizeT num_selected_cpu = 0;
  Value total_weight_cpu = 0;

  if (graph.nodes <= 50)
  {
    printf("CPU Minimum Spanning Tree\n");
  }
  for (std::vector < Edge >::iterator ei = spanning_tree.begin();
       ei != spanning_tree.end(); ++ei)
  {
    if (graph.nodes <= 50)
    {
      // print the edge pairs in the minimum spanning tree
      printf("%ld %ld\n", source(*ei, g), target(*ei, g));
      // printf("  with weight of %f\n", weight[*ei]);
    }
    ++num_selected_cpu;
    total_weight_cpu += weight[*ei];
  }

  // clean up if necessary
  if (edge_pairs) { delete [] edge_pairs; }

  printf("CPU - Computation Complete in %lf msec.\n", elapsed_cpu);
  // printf("CPU - Number of Edges in MST: %d\n", num_selected_cpu);

  return total_weight_cpu;
}

///////////////////////////////////////////////////////////////////////////////
// GPU MST test routines
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Sample test entry
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] parameter Test parameter settings.
 */
template <
  typename VertexId,
  typename SizeT,
  typename Value,
  bool     DEBUG,
  bool     SIZE_CHECK >
void RunTest(MST_Test_Parameter *parameter)
{
  printf("\nMINIMUM SPANNING TREE TEST\n");

  // define the problem data structure for graph primitive
  typedef MSTProblem<
    VertexId,
    SizeT,
    Value,
    true,    // MARK_PREDECESSORS
    false,   // ENABLE_IDEMPOTENCE
    true >   // USE_DOUBLE_BUFFER
    Problem;

  Csr<VertexId, Value, SizeT>* graph =
        (Csr<VertexId, Value, SizeT>*)parameter->graph;
  ContextPtr* context            = (ContextPtr*)parameter -> context;
  std::string partition_method   = parameter -> partition_method;
  int         max_grid_size      = parameter -> max_grid_size;
  int         num_gpus           = parameter -> num_gpus;
  int*        gpu_idx            = parameter -> gpu_idx;
  int         iterations         = parameter -> iterations;
  bool        g_quick            = parameter -> g_quick;
  bool        g_stream_from_host = parameter -> g_stream_from_host;
  double      max_queue_sizing   = parameter -> max_queue_sizing;

  // allocate MST enactor map
  MSTEnactor<
    Problem,
    false,        // INSTRUMENT
    DEBUG,        // DEBUG
    SIZE_CHECK >  // SIZE_CHECK
    mst_enactor(gpu_idx);

  // allocate problem on GPU create a pointer of the MSTProblem type
  Problem * mst_problem = new Problem;

  // host results spaces
  VertexId * h_mst_output = new VertexId[graph->edges];

  // copy data from CPU to GPU initialize data members in DataSlice
  util::GRError(mst_problem->Init(g_stream_from_host, *graph, num_gpus),
    "Problem MST Initialization Failed", __FILE__, __LINE__);

  //
  // perform calculations
  //

  GpuTimer gpu_timer;  // record the kernel running time
  float elapsed_gpu = 0.0f;  // device elapsed running time

  for (int iter = 0; iter < iterations; ++iter)
  {
    // reset values in DataSlice
    util::GRError(mst_problem->Reset(
      mst_enactor.GetFrontierType(), max_queue_sizing),
      "MST Problem Data Reset Failed", __FILE__, __LINE__);

    gpu_timer.Start();

    // launch MST enactor
    util::GRError(mst_enactor.template Enact<Problem>(
      *context, mst_problem, max_grid_size),
      "MST Problem Enact Failed", __FILE__, __LINE__);

    gpu_timer.Stop();
    elapsed_gpu += gpu_timer.ElapsedMillis();
  }

  elapsed_gpu /= iterations;
  printf("GPU - Computation Complete in %lf msec.\n", elapsed_gpu);

  // copy results back to CPU from GPU using Extract
  util::GRError(mst_problem->Extract(h_mst_output),
    "MST Problem Data Extraction Failed", __FILE__, __LINE__);

  if (!g_quick)  // run CPU reference test
  {
    // calculate GPU final number of selected edges
    int num_selected_gpu = 0;
    for (int iter = 0; iter < graph->edges; ++iter)
    {
      num_selected_gpu += h_mst_output[iter];
    }
    // printf("\nGPU - Number of Edges in MST: %d\n", num_selected_gpu);

    // calculate GPU total selected MST weights for validation
    Value total_weight_gpu = 0;
    for (int iter = 0; iter < graph->edges; ++iter)
    {
      total_weight_gpu += h_mst_output[iter] * graph->edge_values[iter];
    }

    // correctness validation
    Value total_weight_cpu = SimpleReferenceMST(graph->edge_values, *graph);
    if (total_weight_cpu == total_weight_gpu)
    {
      // print the edge pairs in the minimum spanning tree
      DisplaySolution(*graph, h_mst_output);
      printf("\nCORRECT.\n");
      std::cout << "CPU Total Weight = " << total_weight_cpu << std::endl;
      std::cout << "GPU Total Weight = " << total_weight_gpu << std::endl;
    }
    else
    {
      printf("INCORRECT.\n");
      std::cout << "CPU Total Weight = " << total_weight_cpu << std::endl;
      std::cout << "GPU Total Weight = " << total_weight_gpu << std::endl;
    }
  }

  // clean up if necessary
  if (mst_problem)  delete     mst_problem;
  if (h_mst_output) delete [] h_mst_output;
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam DEBUG
 *
 * @param[in] parameter Pointer to test parameter settings
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT,
  bool     DEBUG >
void RunTests_size_check(MST_Test_Parameter *parameter)
{
  if (parameter->size_check)
  {
    RunTest <VertexId, Value, SizeT, DEBUG,  true>(parameter);
  }
  else
  {
    RunTest <VertexId, Value, SizeT, DEBUG, false>(parameter);
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] parameter Pointer to test parameter settings
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT >
void RunTests_debug(MST_Test_Parameter *parameter)
{
  if (parameter->debug)
  {
    RunTests_size_check <VertexId, Value, SizeT,  true>(parameter);
  }
  else
  {
    RunTests_size_check <VertexId, Value, SizeT, false>(parameter);
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph    Pointer to the CSR graph we process on.
 * @param[in] args     Reference to the command line arguments.
 * @param[in] num_gpus Number of GPUs.
 * @param[in] context  CudaContext pointer for ModernGPU APIs.
 * @param[in] gpu_idx  GPU index to run algorithm.
 * @param[in] streams  CUDA streams.
 */
template <typename VertexId, typename Value, typename SizeT>
void RunTest(
  Csr<VertexId, Value, SizeT>* graph,
  CommandLineArgs&             args,
  int                          num_gpus,
  ContextPtr*                  context,
  int*                         gpu_idx,
  cudaStream_t*                streams = NULL)
{
  // test graph connectivity because MST only supports fully-connected graph
  struct GRTypes data_t;          // data type structure
  data_t.VTXID_TYPE = VTXID_INT;  // vertex identifier
  data_t.SIZET_TYPE = SIZET_INT;  // graph size type
  data_t.VALUE_TYPE = VALUE_INT;  // attributes type

  struct GRSetup config;          // gunrock configurations
  config.num_devices = num_gpus;  // number of devices
  config.device_list = gpu_idx;   // device used for run
  config.quiet       = true;      // don't print out anything

  struct GRGraph *grapho = (GRGraph*)malloc(sizeof(GRGraph));
  struct GRGraph *graphi = (GRGraph*)malloc(sizeof(GRGraph));

  graphi->num_nodes   = graph->nodes;
  graphi->num_edges   = graph->edges;
  graphi->row_offsets = (void*)&graph->row_offsets[0];
  graphi->col_indices = (void*)&graph->column_indices[0];

  gunrock_cc(grapho, graphi, config, data_t);

  // run test only if the graph is fully-connected
  int* num_cc = (int*)grapho->aggregation;
  if (*num_cc == 1)  // perform minimum spanning tree test
  {
    MST_Test_Parameter *parameter = new MST_Test_Parameter;

    parameter -> Init(args);
    parameter -> graph    = graph;
    parameter -> num_gpus = num_gpus;
    parameter -> context  = context;
    parameter -> gpu_idx  = gpu_idx;
    parameter -> streams  = streams;

    RunTests_debug<VertexId, Value, SizeT>(parameter);
  }
  else  // more than one connected components in the graph
  {
    fprintf(stderr, "Unsupported non-fully connected graph input.\n");
    exit(1);
  }

  if (graphi) free(graphi);
  if (grapho) free(grapho);
}

///////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  if ((argc < 3) || (args.CheckCmdLineFlag("help")))
  {
    Usage();
    return 1;
  }

  int device = 0;
  args.GetCmdLineArgument("device", device);
  mgpu::ContextPtr context = mgpu::CreateCudaDevice(device);

  bool g_undirected = true;  // graph-construction parameters

  std::string graph_type = argv[1];
  int flags = args.ParsedArgc();
  int graph_args = argc - flags - 1;

  if (graph_args < 1)
  {
    Usage();
    return 1;
  }

  //
  // construct graph and perform algorithm
  //

  if (graph_type == "market")
  {
    // matrix-market coordinate-formatted graph file

    // currently support Value type: int, float, double
    typedef int VertexId;  // use as the vertex identifier
    typedef int Value;     // use as the value type
    typedef int SizeT;     // use as the graph size

    // default value for stream_from_host is false
    if (graph_args < 1)
    {
      Usage();
      return 1;
    }

    char * market_filename = (graph_args == 2) ? argv[2] : NULL;

    // buildMarketGraph() reads a .mtx file into CSR data structure
    // template argument = true because the graph has edge values
    Csr<VertexId, Value, SizeT> csr(false);
    if (graphio::BuildMarketGraph<true>(
      market_filename, csr, g_undirected, false) != 0)
    {
      return 1;
    }

    // display input graph with weights
    // csr.DisplayGraph(true);

    //
    // Minimum Spanning Tree only supports undirected, connected graph
    //

    RunTest<VertexId, Value, SizeT>(&csr, args, 1, &context, &device);
  }
  else
  {
    fprintf(stderr, "Unspecified graph type.\n");
    return 1;
  }

  return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
