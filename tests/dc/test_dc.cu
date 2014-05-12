// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_dc.cu
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

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Degree Centrality includes
#include <gunrock/app/dc/dc_enactor.cuh>
#include <gunrock/app/dc/dc_problem.cuh>
//#include <gunrock/app/dc/dc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::dc;

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
	printf("\ntest_top <graph type> <graph type args> [--device=<device_index>] "
		"[--instrumented] [--quick] "
		"[--v]\n"
		"\n"
		"Graph types and args:\n"
		"  market [<file>]\n"
		"    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
		"    edges from stdin (or from the optionally-specified file).\n"
		"  --device=<device_index>  Set GPU device for running the graph primitive.\n"
		"  --instrumented If set then kernels keep track of queue-search_depth\n"
		"  and barrier duty (a relative indicator of load imbalance.)\n"
		"  --quick If set will skip the CPU validation code.\n"
		);
}

/**
 * @brief Displays the top result
 *
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(VertexId *h_node_id, Value *h_degrees, SizeT num_nodes)
{
	// only display top K degrees
	if (num_nodes > 100)
	{
		num_nodes = 100;
	}
	printf("\n-----> Top %d degree nodes: \n", num_nodes);
	printf("[    node_id    |   num_degree  ]\n");
	for (int i = 0; i < num_nodes; ++i)
	{
		printf("%16d %16d\n", h_node_id[i], h_degrees[i]);
	}
}

/**
 * @brief Comparison for the Degree Centrality result
 *
 */
int compareResults()
{
	printf(" Comparing results ...\n");
	return 0;
}

/******************************************************************************
 * Degree Centrality Testing Routines
 *****************************************************************************/
/**
 * @brief A simple CPU-based reference DC implementation.
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
void SimpleReferenceDC(
    Value	*weights,
	const Csr<VertexId, Value, SizeT> &graph)
{
    // Preparation

    // Compute DC using CPU
    CpuTimer cpu_timer; // record the kernel running time  	
	
	cpu_timer.Start();
	
	cpu_timer.Stop();
    	
	float elapsed_cpu = cpu_timer.ElapsedMillis();
	
    printf(" CPU Degree Centrality finished in %lf msec.\n", elapsed_cpu);
    printf(" --- CPU Degree Centrality Complete ---\n");
}

/**
 * @brief Run DC tests
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
void RunTests(
    const Csr<VertexId, Value, SizeT> &graph,
    int max_grid_size,
    int num_gpus,
    mgpu::CudaContext& context)
{
	/* Define the problem data structure for graph primitive */
    typedef DCProblem<
        VertexId,
        SizeT,
        Value> Problem;
	
	/* INSTRUMENT specifies whether we want to keep such statistical data */
    // Allocate DC enactor map 
    DCEnactor<INSTRUMENT> dc_enactor(g_verbose);

    /* Allocate problem on GPU */
    // Create a pointer of the DCProblem type 
	Problem *dc_problem = new Problem;
    	
	/* Copy data from CPU to GPU */
	// Initialize data members in DataSlice 
	util::GRError(dc_problem->Init(
        g_stream_from_host,
        graph,
        num_gpus), "Problem DC Initialization Failed", __FILE__, __LINE__);

    // Perform Degree Centrality
    GpuTimer gpu_timer; // Record the kernel running time 
	
	/* Reset values in DataSlice */
    util::GRError(dc_problem->Reset(dc_enactor.GetFrontierType()), 
    	"DC Problem Data Reset Failed", __FILE__, __LINE__);
    
    gpu_timer.Start();
    util::GRError(dc_enactor.template Enact<Problem>(context, dc_problem, max_grid_size), 
			"DC Problem Enact Failed", __FILE__, __LINE__);
	gpu_timer.Stop();

	float elapsed_gpu = gpu_timer.ElapsedMillis();
	printf(" GPU Degree Centrality finished in %lf msec.\n", elapsed_gpu);
	
	// Copy out results back to CPU from GPU using Extract 
    VertexId	*h_node_id = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
	Value 		*h_degrees = (Value*)malloc(sizeof(Value) * graph.nodes);
	
	util::GRError(dc_problem->Extract(h_node_id, h_degrees), 
		"DC Problem Data Extraction Failed", __FILE__, __LINE__);

	// Display solution
	DisplaySolution(h_node_id, h_degrees, graph.nodes);
		
    // Cleanup
    if (dc_problem) delete dc_problem;

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
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
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
        RunTests<VertexId, Value, SizeT, true>(
            graph,
            max_grid_size,
            num_gpus,
            context);
    } 
    else 
    {
        RunTests<VertexId, Value, SizeT, false>(
            graph,
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
	//srand(0);			// Presently deterministic
	//srand(time(NULL));

	// Parse graph-contruction params
	g_undirected = false;

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

		typedef int VertexId;	// Use as the node identifier type
		typedef int Value;		// Use as the value type
		typedef int SizeT;		// Use as the graph size type
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
		if (graphio::BuildMarketGraph<true>(
			market_filename, 
			csr, 
			g_undirected,
			false) != 0) // no inverse graph
		{
			return 1;
		}

		// display graph	
		// csr.DisplayGraph();
		
		// run gpu tests
		RunTests(csr, args, *context);
	
	}
	else 
	{
		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;
	}

	return 0;
}

/* end */
