// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_mst.cu
 *
 * @brief Simple test driver for computing Minimum Spanning Tree.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// MST includes
#include <gunrock/app/mst/mst_enactor.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <moderngpu.cuh>

// CPU Prim's mst reference
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::mst;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/
bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
  printf(
  "\ntest_mst <graph type> <graph type args> [--device=<device_index>] "
  "[--instrumented] [--quick] "
  "[--v]\n"
  "\n"
  "Graph types and args:\n"
  "  market [<file>]\n"
  "    Reads a Matrix-Market coordinate-format graph of directed/undirected\n"
  "    edges from stdin (or from the optionally-specified file)\n"
  "  --device=<device_index>  Set GPU device for running the graph primitive\n"
  "  --instrumented If set then kernels keep track of queue-search_depth\n"
  "  and barrier duty (a relative indicator of load imbalance)\n"
  "  --quick If set will skip the CPU validation code\n");
}

/**
 * @brief Displays the MST result
 *
 */
template<
  typename VertexId,
  typename Value,
  typename SizeT>
void DisplaySolution(
  const Csr<VertexId, Value, SizeT> &graph, int *mst_output)
{
  VertexId *temp_keys = new VertexId[graph.edges];
  for (int i = 0; i < graph.nodes; ++i)
  {
    for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
    {
      temp_keys[j] = i;
    }
  }

  for (int i = 0; i < graph.edges; ++i)
  {
    if (mst_output[i] == 1)
    {
      std::cout << "parent[" << temp_keys[i] << "] = "
                << graph.column_indices[i] << std::endl;

    }
  }

  if (temp_keys) { delete [] temp_keys; }
}

/******************************************************************************
 * MST Testing Routines
 *****************************************************************************/
/**
 * @brief A simple CPU-based reference MST implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 */
template<
  typename VertexId,
  typename Value,
  typename SizeT>
Value SimpleReferenceMST(
  const Value *edge_values, const Csr<VertexId, Value, SizeT> &graph)
{

  const int num_nodes = graph.nodes;
  const int num_edges = graph.edges;
  printf(" Reference Test: #nodes: %d #edges: %d\n", num_nodes, num_edges);

  // kruskal_min_spanning_tree preparations
  using namespace boost;
  typedef adjacency_list < vecS, vecS, undirectedS,
    no_property, property < edge_weight_t, int > > Graph;
  typedef graph_traits < Graph >::edge_descriptor Edge;
  typedef graph_traits < Graph >::vertex_descriptor Vertex;
  typedef std::pair<int, int> E;

  E *edge_pairs = new E[num_edges];
  int idx = 0;
  for (int i = 0; i < num_nodes; ++i)
  {
    for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
    {
      edge_pairs[idx++] = std::make_pair(i, graph.column_indices[j]);
    }
  }

  Graph g(edge_pairs, edge_pairs + num_edges, edge_values, num_nodes);
  property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
  std::vector < Edge > spanning_tree;

  // compute MST using CPU
  CpuTimer cpu_timer; // record the kernel running time

  cpu_timer.Start();
  kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
  cpu_timer.Stop();

  float elapsed_cpu = cpu_timer.ElapsedMillis();

  SizeT num_selected_cpu = 0;
  Value total_weight_cpu = 0;
  std::cout << "Print the edges in the MST:" << std::endl;
  for (std::vector < Edge >::iterator ei = spanning_tree.begin();
       ei != spanning_tree.end(); ++ei)
  {
    //std::cout << source(*ei, g) << " <--> " << target(*ei, g)
    //  << " with weight of " << weight[*ei]
    //  << std::endl;
    ++num_selected_cpu;
    total_weight_cpu += weight[*ei];
  }

  printf(" CPU - Computation Complete in %lf msec.\n", elapsed_cpu);
  printf(" CPU - Number of Edges in MST: %ld\n", num_selected_cpu);

  return total_weight_cpu;
}

/**
 * @brief Run MST tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph_gpu the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 *
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT,
  bool INSTRUMENT>
void RunTests(
  const Csr<VertexId, Value, SizeT> &graph_gpu,
  const Csr<VertexId, Value, SizeT> &graph_cpu,
  int max_grid_size,
  int num_gpus,
  mgpu::CudaContext& context)
{
  // define the problem data structure for graph primitive
  typedef MSTProblem<
  VertexId,
  SizeT,
  Value,
  true> Problem;

  // INSTRUMENT specifies whether we want to keep such statistical data
  // allocate MST enactor map
  MSTEnactor<INSTRUMENT> mst_enactor(g_verbose);

  // allocate problem on GPU create a pointer of the MSTProblem type
  Problem *mst_problem = new Problem;

  // malloc host results spaces
  VertexId *h_mst_output = (VertexId*)malloc(sizeof(VertexId) * graph_gpu.edges);

  // copy data from CPU to GPU initialize data members in DataSlice
  util::GRError(mst_problem->Init(
    g_stream_from_host,
    graph_gpu,
    num_gpus),
    "Problem MST Initialization Failed",
    __FILE__, __LINE__);

  // perform MST
  GpuTimer gpu_timer; // Record the kernel running time

  // reset values in DataSlice
  util::GRError(mst_problem->Reset(mst_enactor.GetFrontierType()),
    "MST Problem Data Reset Failed", __FILE__, __LINE__);

  gpu_timer.Start();

  // launch MST enactor
  util::GRError(mst_enactor.template Enact<Problem>(
    context,
    mst_problem,
    max_grid_size),
    "MST Problem Enact Failed", __FILE__, __LINE__);

  gpu_timer.Stop();

  float elapsed_gpu = gpu_timer.ElapsedMillis();
  printf(" GPU - Computation Complete in %lf msec.\n", elapsed_gpu);

  // copy results back to CPU from GPU using Extract
  util::GRError(mst_problem->Extract(h_mst_output),
    "MST Problem Data Extraction Failed", __FILE__, __LINE__);

  // display computed results
  //DisplaySolution(graph_gpu, h_mst_output);

  // calculate gpu final number of selected edges
  int num_selected_gpu = 0;
  for (int iter = 0; iter < graph_gpu.edges; ++iter)
  {
    num_selected_gpu += h_mst_output[iter];
  }
  printf(" GPU - Number of Edges in MST: %d\n", num_selected_gpu);

  // calculate gpu total selected mst weight for validation
  Value total_weight_gpu = 0;
  for (int iter = 0; iter < graph_gpu.edges; ++iter)
  {
    total_weight_gpu += h_mst_output[iter] * graph_gpu.edge_values[iter];
  }

  // validation
  Value total_weight_cpu = SimpleReferenceMST(graph_cpu.edge_values, graph_cpu);
  if (total_weight_cpu == total_weight_gpu)
  {
    printf("CORRECT.\n");
  }
  else
  {
    printf("INCORRECT. \nCPU Computed Total Weight = %d\n"
      "GPU Computed Total Weight = %d\n", total_weight_cpu, total_weight_gpu);
  }

  // clean up id neccessary
  if (mst_problem)  { delete mst_problem; }
  if (h_mst_output) { free(h_mst_output); }

  cudaDeviceSynchronize();
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph_gpu the CSR graph we process on
 * @param[in] graph_cpu the CSR graph used for reference
 * @param[in] args Reference to the command line arguments
 * @param[in] modern gpu cuda context
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT>
void RunTests(
  const Csr<VertexId, Value, SizeT> &graph_gpu,
  const Csr<VertexId, Value, SizeT> &graph_cpu,
  CommandLineArgs                   &args,
  mgpu::CudaContext&                context)
{
  bool instrumented  = false; //!< do not collect instrumentation from kernels
  int  max_grid_size = 0;     //!< maximum grid size (up to the enactor)
  int  num_gpus      = 1;     //!< number of GPUs for multi-gpu enactor to use

  instrumented = args.CheckCmdLineFlag("instrumented");

  g_quick = args.CheckCmdLineFlag("quick");
  g_verbose = args.CheckCmdLineFlag("v");

  if (instrumented)
  {
    RunTests<VertexId, Value, SizeT, true>(
      graph_gpu,
      graph_cpu,
      max_grid_size,
      num_gpus,
      context);
  }
  else
  {
    RunTests<VertexId, Value, SizeT, false>(
      graph_gpu,
      graph_cpu,
      max_grid_size,
      num_gpus,
      context);
  }
}

/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  if ((argc < 2) || (args.CheckCmdLineFlag("help")))
  {
    Usage();
    return 1;
  }

  //DeviceInit(args);
  //cudaSetDeviceFlags(cudaDeviceMapHost);
  int dev = 0;
  args.GetCmdLineArgument("device", dev);
  mgpu::ContextPtr context = mgpu::CreateCudaDevice(dev);
  //srand(0); // Presently deterministic
  //srand(time(NULL));

  // Parse graph-contruction params
  g_undirected = true;

  std::string graph_type = argv[1];
  int flags = args.ParsedArgc();
  int graph_args = argc - flags - 1;

  if (graph_args < 1)
  {
    Usage();
    return 1;
  }

  //
  // Construct graph and perform search(es)
  //

  if (graph_type == "market")
  {

    // Matrix-market coordinate-formatted graph file

    typedef int VertexId; //!< Use as the node identifier type
    typedef int Value;    //!< Use as the value type
    typedef int SizeT;    //!< Use as the graph size type

    // default value for stream_from_host is false
    if (graph_args < 1)
    {
      Usage();
      return 1;
    }

    char *market_filename = (graph_args == 2) ? argv[2] : NULL;

    // buildMarketGraph() reads a mtx file into CSR data structure
    // Template argumet = true because the graph has edge values
    Csr<VertexId, Value, SizeT> csr_gpu(false);
    if (graphio::BuildMarketGraph<true>(
      market_filename,
      csr_gpu,
      g_undirected,
      false) != 0) { return 1; }

    // boost prim's mst algorithm requires directed graph input
    Csr<VertexId, Value, SizeT> csr_cpu(false);
    if (graphio::BuildMarketGraph<true>(
      market_filename,
      csr_cpu,
      !g_undirected,
      false) != 0) { return 1; }

    // display graph
    //csr_gpu.DisplayGraph();
    //csr_cpu.DisplayGraph();

    // run gpu tests
    RunTests(csr_gpu, csr_cpu, args, *context);

  }
  else
  {
    // unknown graph type
    fprintf(stderr, "Unspecified graph type\n");
    return 1;
  }

  return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: