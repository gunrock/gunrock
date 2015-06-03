// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

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

// Graph construction utilities
#include <gunrock/graphio/market.cuh>

// MST includes
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

////////////////////////////////////////////////////////////////////////////////
// defines, constants, globals
bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

////////////////////////////////////////////////////////////////////////////////
// housekeeping routines
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
    "  --device=<device_index> Set GPU device for running the graph primitive\n"
    "  --instrumented If set then kernels keep track of queue-search_depth\n"
    "  and barrier duty (a relative indicator of load imbalance)\n"
    "  --quick If set will skip the CPU validation code\n"
    "  --v If set will enable DEBUG mode\n\n"
    " --------------------------------------------------------------\n"
    "  To make sure two graphs have same weight value for each edge \n"
    "  we have to change ll_value = rand()%%64 in market.cuh file to \n"
    "  some NON-RANDOM value if the original graph does NOT contain \n"
    "  weight per edge. Note it only support FULLY-CONNECTED graphs \n"
    " --------------------------------------------------------------\n");
}

/**
 * @brief Displays the MST result
 *
 */
////////////////////////////////////////////////////////////////////////////////
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(const Csr<VertexId, Value, SizeT> &graph, int *mst_output)
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

/**
 * @brief A simple CPU-based reference MST implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] edge_values weight value associated with per edge
 * @param[in] graph reference to the CSR graph we process on
 *
 *  \return long long int variable which indicates the total weight of the graph
 */
////////////////////////////////////////////////////////////////////////////////
template<typename VertexId, typename Value, typename SizeT>
long long int SimpleReferenceMST(
  const Value *edge_values, const Csr<VertexId, Value, SizeT> &graph)
{
  printf("\nREFERENCE TEST\n");

  // Kruskal minimum spanning tree preparations
  using namespace boost;
  typedef adjacency_list < vecS, vecS, undirectedS,
    no_property, property < edge_weight_t, int > >  Graph;
  typedef graph_traits < Graph >::edge_descriptor   Edge;
  typedef graph_traits < Graph >::vertex_descriptor Vertex;
  typedef std::pair<int, int> E;

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
  SizeT         num_selected_cpu = 0;
  long long int total_weight_cpu = 0;

  if (graph.nodes <= 50) printf("CPU Minimum Spanning Tree\n");
  for (std::vector < Edge >::iterator ei = spanning_tree.begin();
       ei != spanning_tree.end(); ++ei)
  {
    if (graph.nodes <= 50)
    {
      // print the edge pairs in the minimum spanning tree
      printf("%ld %ld\n", source(*ei, g), target(*ei, g));
      // printf("  with weight of %d\n", weight[*ei]);
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

/**
 * @brief Run MST tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph the CSR graph we send to GPU to process
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 *
 */
////////////////////////////////////////////////////////////////////////////////
template <typename VertexId, typename Value, typename SizeT, bool INSTRUMENT>
void RunTests(
  const Csr<VertexId, Value, SizeT> &graph,
  int max_grid_size,
  int num_gpus,
  mgpu::CudaContext& context)
{
  printf("\nMINIMUM SPANNING TREE TEST\n");

  // define the problem data structure for graph primitive
  typedef MSTProblem<VertexId, SizeT, Value, true> Problem;

  // INSTRUMENT specifies whether we want to keep such statistical data
  // allocate MST enactor map
  MSTEnactor<INSTRUMENT> mst_enactor(g_verbose);

  // allocate problem on GPU create a pointer of the MSTProblem type
  Problem * mst_problem = new Problem;

  // host results spaces
  VertexId * h_mst_output = new VertexId[graph.edges];

  // copy data from CPU to GPU initialize data members in DataSlice
  util::GRError(mst_problem->Init(g_stream_from_host, graph, num_gpus),
    "Problem MST Initialization Failed", __FILE__, __LINE__);

  // reset values in DataSlice
  util::GRError(mst_problem->Reset(mst_enactor.GetFrontierType()),
    "MST Problem Data Reset Failed", __FILE__, __LINE__);

  // perform MST
  GpuTimer gpu_timer; // record the kernel running time

  gpu_timer.Start();

  // launch MST Enactor
  util::GRError(mst_enactor.template Enact<Problem>(
    context, mst_problem, max_grid_size),
    "MST Problem Enact Failed", __FILE__, __LINE__);

  gpu_timer.Stop();

  float elapsed_gpu = gpu_timer.ElapsedMillis();
  printf("GPU - Computation Complete in %lf msec.\n", elapsed_gpu);

  // copy results back to CPU from GPU using Extract
  util::GRError(mst_problem->Extract(h_mst_output),
    "MST Problem Data Extraction Failed", __FILE__, __LINE__);

  if (!g_quick) // run CPU reference test
  {
    // calculate GPU final number of selected edges
    int num_selected_gpu = 0;
    for (int iter = 0; iter < graph.edges; ++iter)
    {
      num_selected_gpu += h_mst_output[iter];
    }
    // printf("\nGPU - Number of Edges in MST: %d\n", num_selected_gpu);

    // calculate GPU total selected MST weights for validation
    long long int total_weight_gpu = 0;
    for (int iter = 0; iter < graph.edges; ++iter)
    {
      total_weight_gpu += h_mst_output[iter] * graph.edge_values[iter];
    }

    // correctness validation
    long long int total_weight_cpu =
      SimpleReferenceMST(graph.edge_values, graph);
    if (total_weight_cpu == total_weight_gpu)
    {
      // print the edge pairs in the minimum spanning tree
      DisplaySolution(graph, h_mst_output);
      printf("\nCORRECT.\n");
    }
    else
    {
      printf("INCORRECT. \n"
             "CPU Computed Total Weight = %lld\n"
             "GPU Computed Total Weight = %lld\n",
             total_weight_cpu, total_weight_gpu);
    }
  }

  // clean up if necessary
  if (mst_problem)  delete     mst_problem;
  if (h_mst_output) delete [] h_mst_output;

  cudaDeviceSynchronize();
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] context modern GPU CUDA context
 */
template <typename VertexId, typename Value, typename SizeT>
void RunTests(
  const Csr<VertexId, Value, SizeT> &graph,
  CommandLineArgs                   &args,
  mgpu::CudaContext&                context)
{
  bool instrumented  = false; // do not collect instrumentation from kernels
  int  max_grid_size = 0;     // maximum grid size (up to the enactor)
  int  num_gpus      = 1;     // number of GPUs for multi-gpu enactor to use
  g_quick            = false; // Whether or not to skip ref validation

  instrumented = args.CheckCmdLineFlag("instrumented");

  g_quick = args.CheckCmdLineFlag("quick");
  g_verbose = args.CheckCmdLineFlag("v");

  if (instrumented)
  {
    RunTests<VertexId, Value, SizeT, true>(
      graph, max_grid_size, num_gpus, context);
  }
  else
  {
    RunTests<VertexId, Value, SizeT, false>(
      graph, max_grid_size, num_gpus, context);
  }
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  if ((argc < 3) || (args.CheckCmdLineFlag("help")))
  {
    Usage();
    return 1;
  }

  int dev = 0;
  args.GetCmdLineArgument("device", dev);
  mgpu::ContextPtr context = mgpu::CreateCudaDevice(dev);

  // parse graph-construction parameters
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
  // construct graph and perform algorithm
  //

  if (graph_type == "market")
  {

    // matrix-market coordinate-formatted graph file

    typedef int VertexId; // use as the vertex identifier type
    typedef int Value;    // use as the value type
    typedef int SizeT;    // use as the graph size type

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
      market_filename,
      csr,
      g_undirected,
      false) != 0) { return 1; }

    // display graph
    // csr.DisplayGraph();

    /***************************************************************
    * To make sure two graphs have same weight value for each edge *
    * we have to change ll_value = rand()%64 in market.cuh file to *
    * some NON-RANDOM value if the original graph does NOT contain *
    * weight per edge. Note it only support FULLY-CONNECTED graphs *
    ***************************************************************/

    // run GPU tests
    RunTests(csr, args, *context);

  }
  else
  {
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
