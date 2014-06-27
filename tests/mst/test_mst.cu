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
 * @brief Simple test driver program for computing Pagerank.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Utilities and Correctness-checking
#include <gunrock/util/test_utils.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// MST includes
#include <gunrock/app/mst/mst_enactor.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advanced/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <moderngpu.cuh>

// CPU Prim's mst reference
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

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
  printf("\ntest_mst <graph type> <graph type args> [--device=<device_index>] "
	 "[--instrumented] [--quick] "
	 "[--v]\n"
	 "\n"
	 "Graph types and args:\n"
	 "  market [<file>]\n"
	 "    Reads a Matrix-Market coordinate-formatted graph\n"
	 "    edges from stdin (or from the optionally-specified file).\n"
	 "  --device=<device_index>  Set device for running graph primitive.\n"
	 "  --instrumented If set, kernels keep track of queue-search_depth\n"
	 "  and barrier duty (a relative indicator of load imbalance.)\n"
	 "  --quick If set will skip the CPU validation code.\n"
	 );
}

/**
 * @brief Displays the MST result
 *
 */
template<typename Value, typename SizeT>
void DisplaySolution()
{
  //TODOf
}

/**
 * @brief Comparison for the MST result
 *
 */
int compareResults()
{
  printf(" Comparing results ...\n");
  return 0;
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
void SimpleReferenceMST(const Csr<VertexId, Value, SizeT> &graph)
{
  printf("\n --- CPU Reference Test ---\n");

  //Preparation
  using namespace boost;
  typedef adjacency_list < vecS, vecS, undirectedS,
			   property<vertex_distance_t, int>,
			   property < edge_weight_t, int > > Graph;

  typedef std::pair < int, int >E;
  const int num_nodes = graph.nodes;
  const int num_edges = graph.edges;

  printf(" Number of nodes: %4d, Number of edges: %4d\n", num_nodes, num_edges);
  E *edge_pairs = new E[num_edges];
  int idx = 0;

  for (int i = 0; i < num_nodes; ++i)
  {
    for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
    {
      edge_pairs[idx++] = std::make_pair(i, graph.column_indices[j]);
    }
  }

  // original total weights
  int weight_sum = 0;
  for (int edgeIter = 0; edgeIter < num_edges; ++edgeIter)
  {
    weight_sum += weights[edgeIter];
  }
  //printf(" Original Total Weights: %d\n", weight_sum);

  Graph g(edge_pairs, edge_pairs + num_edges, weights, num_nodes);
  property_map< Graph, edge_weight_t >::type weightmap = get(edge_weight, g);
  std::vector < graph_traits < Graph >::vertex_descriptor > p(num_vertices(g));

  typedef graph_traits<Graph>::edge_iterator edge_iterator;

  std::pair<edge_iterator, edge_iterator> ei = edges(g);

  /*
  // display read in graph
  for (edge_iterator edge_iter = ei.first; edge_iter != ei.second; ++edge_iter)
  {
    std::cout << "(" << source(*edge_iter, g) << ", " << target(*edge_iter, g) << ")\n";
  }
  */

  // Compute MST using CPU
  CpuTimer cpu_timer; // record the kernel running time

  cpu_timer.Start();
  prim_minimum_spanning_tree(g, &p[0]);
  cpu_timer.Stop();

  float elapsed_cpu = cpu_timer.ElapsedMillis();

  printf(" CPU MST finished in %lf msec.\n", elapsed_cpu);
  /*
  // display mst results
  for (std::size_t i = 0; i != p.size(); ++i)
  {
    if (p[i] != i)
    {
      std::cout << "parent[" << i << "] = " << p[i] << std::endl;
    }
    else
    {
      std::cout << "parent[" << i << "] = no parent" << std::endl;
    }
  }
  */

  printf(" Number of edges selected - %d\n", p.size() - 1);
  printf(" --- CPU MST Complete ---\n");
}

/**
 * @brief Run MST tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
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
void RunTests(const Csr<VertexId, Value, SizeT> &graph,
	      int max_grid_size,
	      int num_gpus,
	      mgpu::CudaContext& context)
{
  /* Define the problem data structure for graph primitive */
  typedef MSTProblem<
    VertexId,
    SizeT,
    Value,
    true> Problem;

  /* INSTRUMENT specifies whether we want to keep such statistical data */
  // Allocate MST enactor map
  MSTEnactor<INSTRUMENT> mst_enactor(g_verbose);

  /* Allocate problem on GPU */
  // Create a pointer of the MSTProblem type
  Problem *mst_problem = new Problem;

  /* Copy data from CPU to GPU */
  // Initialize data members in DataSlice
  util::GRError(mst_problem->Init(g_stream_from_host,
				  graph,
				  num_gpus),
		"Problem MST Initialization Failed", __FILE__, __LINE__);

  // Perform MST
  GpuTimer gpu_timer; // Record the kernel running time

  /* Reset values in DataSlice */
  util::GRError(mst_problem->Reset(mst_enactor.GetFrontierType()),
		"MST Problem Data Reset Failed", __FILE__, __LINE__);

  gpu_timer.Start();

  util::GRError(mst_enactor.template Enact<Problem>(context,
						    mst_problem,
						    max_grid_size),
		"MST Problem Enact Failed", __FILE__, __LINE__);

  gpu_timer.Stop();

  float elapsed_gpu = gpu_timer.ElapsedMillis();
  printf(" GPU MST finished in %lf msec.\n", elapsed_gpu);

  /* Copy out results back to CPU from GPU using Extract */
  // TODO: write the extract function
  // util::GRError(csr_problem->Extract(h_result),
  //"MST Problem Data Extraction Failed", __FILE__, __LINE__);

  /* Display solution*/
  // DisplaySolution()

  /* Cleanup */
  if (mst_problem) delete mst_problem;
  // TODO host arrays.

  cudaDeviceSynchronize();
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT>
void RunTests(Csr<VertexId, Value, SizeT> &graph,
	      CommandLineArgs &args,
	      mgpu::CudaContext& context)
{
  bool instrumented = false;
  int max_grid_size = 0;
  int num_gpus = 1;

  instrumented = args.CheckCmdLineFlag("instrumented");

  g_quick = args.CheckCmdLineFlag("quick");
  g_verbose = args.CheckCmdLineFlag("v");

  if (instrumented)
  {
    RunTests<VertexId, Value, SizeT, true>(graph,
					   max_grid_size,
					   num_gpus,
					   context);
  }
  else
  {
    RunTests<VertexId, Value, SizeT, false>(graph,
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

  int dev = 0;
  args.GetCmdLineArgument("device", dev);
  mgpu::ContextPtr context = mgpu::CreateCudaDevice(dev);

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
    /* Matrix-market coordinate-formatted graph file */

    typedef int VertexId; //!< Use as the node identifier type
    typedef int Value;    //!< Use as the value type
    typedef int SizeT;    //!< Use as the graph size type

    Csr<VertexId, Value, SizeT> csr(false);

    /* Default value for stream_from_host is false */
    if (graph_args < 1)
    {
      Usage();
      return 1;
    }

    char *market_filename = (graph_args == 2) ? argv[2] : NULL;

    /* BuildMarketGraph() reads a mtx file into CSR data structure */
    // Template argumet = true because the graph has edge weights
    if (graphio::BuildMarketGraph<true>(market_filename,
					csr,
					g_undirected,
					false) != 0) // no inverse graph
    {
      return 1;
    }

    // display graph
    csr.DisplayGraph();

    // run gpu tests
    RunTests(csr, args, *context);

    /* run cpu reference test */
    // build a directed graph required by cpu reference computing
    Csr<VertexId, Value, SizeT> csr2(false);
    if (graphio::BuildMarketGraph<true>(market_filename,
					csr2,
					true,
					false) != 0)
    { return 1; }

    //csr2.DisplayGraph();

    SimpleReferenceMST(csr2.edge_values, csr2);

    // verify results using compareResults() function
  }
  else
  {
    // Unknown graph type
    fprintf(stderr, "Unspecified graph type.\n");
    return 1;
  }

  return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
