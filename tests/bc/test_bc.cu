// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bc.cu
 *
 * @brief Simple test driver program for BFS.
 */

#include <stdio.h> 
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BC includes
#include <gunrock/app/bc/bc_enactor.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bc;


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
 printf("\ntest_bc <graph type> <graph type args> [--device=<device_index>] "
        "[--instrumented] [--src=<source index>] [--quick] "
        "[--num_gpus=<gpu number>] [--queue-sizing=<scale factor>]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "--src=<source index>: When source index is -1, compute BC value for each\n"
        "node. Otherwise, debug the delta value for one node\n"
        );
 }

 /**
  * Displays the BC result (sigma value and BC value)
  */
 template<typename Value, typename SizeT>
 void DisplaySolution(Value *sigmas, Value *bc_values, SizeT nodes)
 {
     if (nodes < 20) {
         printf("[");
         for (SizeT i = 0; i < nodes; ++i) {
             PrintValue(i);
             printf(":");
             PrintValue(sigmas[i]);
             printf(",");
             PrintValue(bc_values[i]);
             printf(" ");
         }
         printf("]\n");
     }
 }

 /**
  * Performance/Evaluation statistics
  */

 struct Statistic
 {
    double mean;
    double m2;
    int count;

    Statistic() : mean(0.0), m2(0.0), count(0) {}

    /**
     * Updates running statistic, returning bias-corrected sample variance.
     * Online method as per Knuth.
     */
    double Update(double sample)
    {
        count++;
        double delta = sample - mean;
        mean = mean + (delta / count);
        m2 = m2 + (delta * (sample - mean));
        return m2 / (count - 1);                //bias-corrected
    }
};

/******************************************************************************
 * BC Testing Routines
 *****************************************************************************/

 /**
  * A simple CPU-based reference BC ranking implementation.
  */
 template<
    typename VertexId,
    typename Value,
    typename SizeT>
void RefCPUBC(
    const Csr<VertexId, Value, SizeT>       &graph,
    Value                                   *sigmas,
    Value                                   *bc_values,
    VertexId                                src)
{
    //
    //Perform BC
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printf("CPU BC finished in %lf msec.", elapsed);
}

/**
 * Run tests
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT>
void RunTests(
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId src,
    int max_grid_size,
    int num_gpus,
    double max_queue_sizing)
{
    typedef BCProblem<
        VertexId,
        SizeT,
        Value,
        io::ld::cg,
        io::ld::NONE,
        io::ld::NONE,
        io::ld::cg,
        io::ld::NONE,
        io::st::cg> Problem;

    typedef ForwardFunctor<
        VertexId,
        SizeT,
        Value,
        Problem> FFunctor;
    typedef BackwardFunctor<
        VertexId,
        SizeT,
        Value,
        Problem> BFunctor;


        // Allocate host-side array (for both reference and gpu-computed results)
        Value       *reference_bc_values        = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value       *reference_sigmas           = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value       *h_sigmas                   = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value       *h_bc_values                = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value       *reference_check_sigmas     = (g_quick) ? NULL : reference_sigmas;
        Value       *reference_check_bc_values  = (g_quick) ? NULL : reference_bc_values;

        // Allocate BC enactor map
        BCEnactor<INSTRUMENT> bc_enactor(g_verbose);

        printf("edge: %d\n", graph.edges);

        // Allocate problem on GPU
        Problem *csr_problem = new Problem;
        if (csr_problem->Init(
            g_stream_from_host,
            graph.nodes,
            graph.edges,
            graph.row_offsets,
            graph.column_indices,
            num_gpus)) exit(1);

        //
        // Compute reference CPU BC solution for source-distance
        //
        if (reference_check_bc_values != NULL)
        {
            printf("compute ref value\n");
            RefCPUBC(
                    graph,
                    reference_check_sigmas,
                    reference_check_bc_values,
                    src);
            printf("\n");
        }

        cudaError_t         retval = cudaSuccess;

        // Perform BFS
        GpuTimer gpu_timer;

        VertexId start_src;
        VertexId end_src;
        if (src == -1)
        {
            start_src = 0;
            end_src = graph.nodes;
        }
        else
        {
            start_src = src;
            end_src = src+1;
        }


        for (VertexId i = start_src; i < end_src; ++i)
        {
            if (retval = csr_problem->Reset(i, bc_enactor.GetFrontierType(), max_queue_sizing)) exit(1);
            gpu_timer.Start();
            if (retval = bc_enactor.template Enact<Problem, FFunctor, BFunctor>(csr_problem, i, max_grid_size)) exit(1);
            gpu_timer.Stop();

            if (retval && (retval != cudaErrorInvalidDeviceFunction)) {
                exit(1);
            }
        }

        float elapsed = gpu_timer.ElapsedMillis();

        // Copy out results
        if (csr_problem->Extract(h_sigmas, h_bc_values)) exit(1);

        // Verify the result
        if (reference_check_bc_values != NULL) {
            printf("Validity: ");
            CompareResults(h_bc_values, reference_check_bc_values, graph.nodes, true);
        }
        
        // Display Solution
        DisplaySolution(h_sigmas, h_bc_values, graph.nodes);


        // Cleanup
        if (csr_problem) delete csr_problem;
        if (reference_bc_values) free(reference_bc_values);
        if (reference_sigmas) free(reference_sigmas);
        if (h_sigmas) free(h_sigmas);
        if (h_bc_values) free(h_bc_values);

        cudaDeviceSynchronize();
}

template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    CommandLineArgs &args)
{
    VertexId            src                 = -1;           // Use whatever the specified graph-type's default is
    std::string         src_str;
    bool                instrumented        = false;        // Whether or not to collect instrumentation from kernels
    int                 max_grid_size       = 0;            // maximum grid size (0: leave it up to the enactor)
    int                 num_gpus            = 1;            // Number of GPUs for multi-gpu enactor to use
    double              max_queue_sizing    = 1.3;          // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).

    instrumented = args.CheckCmdLineFlag("instrumented");
    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty()) {
        src = 0;
    } else {
        args.GetCmdLineArgument("src", src);
    }

    g_quick = args.CheckCmdLineFlag("quick");
    args.GetCmdLineArgument("num-gpus", num_gpus);
    args.GetCmdLineArgument("queue-sizing", max_queue_sizing);
    g_verbose = args.CheckCmdLineFlag("v");

    if (instrumented) {
            RunTests<VertexId, Value, SizeT, true>(
                graph,
                src,
                max_grid_size,
                num_gpus,
                max_queue_sizing);
    } else {
            RunTests<VertexId, Value, SizeT, false>(
                graph,
                src,
                max_grid_size,
                num_gpus,
                max_queue_sizing);
    }

}



/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
		Usage();
		return 1;
	}

	DeviceInit(args);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	//srand(0);									// Presently deterministic
	//srand(time(NULL));

	// Parse graph-contruction params
	g_undirected = false;

	std::string graph_type = argv[1];
	int flags = args.ParsedArgc();
	int graph_args = argc - flags - 1;

	if (graph_args < 1) {
		Usage();
		return 1;
	}
	
	//
	// Construct graph and perform search(es)
	//

	if (graph_type == "market") {

		// Matrix-market coordinate-formatted graph file

		typedef int VertexId;							// Use as the node identifier type
		typedef float Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		Csr<VertexId, Value, SizeT> csr(false);         // default value for stream_from_host is false

		if (graph_args < 1) { Usage(); return 1; }
		char *market_filename = (graph_args == 2) ? argv[2] : NULL;
		if (graphio::BuildMarketGraph<false>(
			market_filename, 
			csr, 
			g_undirected) != 0) 
		{
			return 1;
		}

        csr.DisplayGraph();
        fflush(stdout);

		// Run tests
		RunTests(csr, args);

	} else {

		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;

	}

	return 0;
}
