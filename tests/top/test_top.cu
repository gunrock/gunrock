// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_top.cu
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

// top includes
#include <gunrock/app/top/top_enactor.cuh>
#include <gunrock/app/top/top_problem.cuh>
#include <gunrock/app/top/top_functor.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <moderngpu.cuh>

// CPU Prim's top reference
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::top;

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
template<typename Value, typename SizeT>
void DisplaySolution()
{	

}

/**
 * @brief Comparison for the TOP result
 *
 */
int compareResults()
{
	printf(" Comparing results ...\n");
	return 0;
}

/******************************************************************************
 * TOP Testing Routines
 *****************************************************************************/
/**
 * @brief A simple CPU-based reference TOP implementation.
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
void SimpleReferenceTOP(
    Value	*weights,
	const Csr<VertexId, Value, SizeT> &graph)
{
    // Preparation

    // Compute TOP using CPU
    CpuTimer cpu_timer; // record the kernel running time  	
	
	cpu_timer.Start();
	
	cpu_timer.Stop();
    	
	float elapsed_cpu = cpu_timer.ElapsedMillis();
	
    printf(" CPU TOP finished in %lf msec.\n", elapsed_cpu);
    printf(" --- CPU TOP Done ---\n");
}

/**
 * @brief Run TOP tests
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
    typedef TOPProblem<
        VertexId,
        SizeT,
        Value> Problem;
	
	/* INSTRUMENT specifies whether we want to keep such statistical data */
    // Allocate TOP enactor map 
    TOPEnactor<INSTRUMENT> top_enactor(g_verbose);

    /* Allocate problem on GPU */
    // Create a pointer of the TOPProblem type 
	Problem *top_problem = new Problem;
    	
	/* Copy data from CPU to GPU */
	// Initialize data members in DataSlice 
	util::GRError(top_problem->Init(
        g_stream_from_host,
        graph,
        num_gpus), "Problem TOP Initialization Failed", __FILE__, __LINE__);

    // Perform TOP
    GpuTimer gpu_timer; // Record the kernel running time 
	
	/* Reset values in DataSlice */
    util::GRError(top_problem->Reset(top_enactor.GetFrontierType()), 
    	"top Problem Data Reset Failed", __FILE__, __LINE__);
    
    gpu_timer.Start();
    util::GRError(top_enactor.template Enact<Problem>(context, top_problem, max_grid_size), 
	    "top Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();

    float elapsed_gpu = gpu_timer.ElapsedMillis();
	printf(" GPU top finished in %lf msec.\n", elapsed_gpu);
	
    // Copy out results back to CPU from GPU using Extract 
    // TODO: write the extract function
    // util::GRError(csr_problem->Extract(h_result), 
	//	"top Problem Data Extraction Failed", __FILE__, __LINE__);

	// Display solution
	// DisplaySolution()
		
    // Cleanup
    if (top_problem) delete top_problem;

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

		typedef int VertexId;	// Use as the node identifier type
		typedef int Value;	// Use as the value type
		typedef int SizeT;	// Use as the graph size type
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
		csr.DisplayGraph();
		
		// run gpu tests
		RunTests(csr, args, *context);
		
		// verify results using compareResults() function 
		// int result = compareResults();
		// printf(" Verifying results ... %s\n", (result == 1) ? "Success!" : "Failed!");
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
