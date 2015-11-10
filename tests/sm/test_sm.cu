// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sm_test.cu
 * @brief Simple test driver program
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// utilities for correctness checking
#include <gunrock/util/test_utils.cuh>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// primitive-specific headers include
#include <gunrock/app/sm/sm_enactor.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/app/sm/sm_functor.cuh>

// gunrock abstraction graph operators
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sm;

// ----------------------------------------------------------------------------
// Defines, constants, globals
// ----------------------------------------------------------------------------

bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

// ----------------------------------------------------------------------------
// Housekeeping Routines
// ----------------------------------------------------------------------------
void Usage() {
    printf(
        " sm_test <graph type> <graph type args> [--undirected] [--quick] \n"
        " [--device=<device_index>] [--instrumented] [--iteration-num=<num>]\n"
        " [--v] [--traversal-mode=<0|1>] [--queue-sizing=<scale factor>]\n"
        "Graph types and arguments:\n"
        "  market <file>\n"
        "    Reads a Matrix-Market coordinate-formatted graph,\n"
        "    edges from STDIN (or from the optionally-specified file)\n"
        "  --device=<device_index>   Set GPU device to run. [Default: 0]\n"
        "  --undirected              Convert the graph to undirected\n"
        "  --instrumented            Keep kernels statics [Default: Disable]\n"
        "                            total_queued, search_depth and avg_duty\n"
        "                            (a relative indicator of load imbalance)\n"
        "  --quick                   Skip the CPU validation [Default: false]\n"
        "  --queue-sizing=<factor>   Allocates a frontier queue sized at: \n"
        "                            (graph-edges * <factor>) [Default: 1.0]\n"
        "  --v                       Print verbose per iteration debug info\n"
        "  --iteration-num=<number>  Number of tests to run [Default: 1]\n"
        "  --traversal-mode=<0 | 1>  Set strategy, 0 for Load-Balanced,\n"
        "                            1 for Dynamic-Cooperative\n"
        "                            [Default: according to topology]\n");
}

/**
 * @brief Displays subgraph matching results
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 */
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(const Csr<VertexId, Value, SizeT> &graph, bool *h_c_set) {
    printf("-- display solution: (currently missing)\n");
    // TODO: code to print out results
   
}


/**
 * @brief Performance / Evaluation statistics
 */
struct Stats {
    const char *name;
    Statistic num_iterations;
    Stats() : name(NULL), num_iterations() {}
    explicit Stats(const char *name) : name(name), num_iterations() {}
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] stats Reference to the Stats object
 * @param[in] graph Reference to the CSR graph we process on
 */
template<typename VertexId, typename SizeT, typename Value>
void DisplayStats(const Stats &stats, const Csr<VertexId, Value, SizeT> &graph,
                  const float elapsed, const long long iterations) {
    printf("[%s] finished.\n", stats.name);
    printf("elapsed: %.4f ms\n", elapsed);
    printf("num_iterations: %lld\n", iterations);
    // TODO: code to print statistics
}

// ----------------------------------------------------------------------------
// Testing Routines
// ----------------------------------------------------------------------------

/**
 * @brief A simple CPU-based reference implementation.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph we process on
 */
template<typename VertexId, typename SizeT, typename Value>
void SimpleReference(const Csr<VertexId, Value, SizeT> &graph_query, 
		     const Csr<VertexId, Value, SizeT> &graph_data, 
		     bool *h_c_set) {
    // initialization

    // perform calculation

    CpuTimer cpu_timer;
    cpu_timer.Start();

    // TODO: CPU validation code here

    cpu_timer.Stop();

    float cpu_elapsed = cpu_timer.ElapsedMillis();
    printf("CPU reference finished in %lf ms.\n\n", cpu_elapsed);
}

/**
 * @brief Subgraph Matching test
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] max_queue_sizing Scaling factor used in edge mapping
 * @param[in] iterations Number of iterations for running the test
 * @param[in] traversal_mode Strategy: Load-balanced or Dynamic cooperative
 * @param[in] numer of nodes in the query graph
 * @param[in] context CudaContext pointer for ModernGPU APIs
 *
 */
template<typename VertexId, typename SizeT, typename Value, bool INSTRUMENT>
void RunTest(
    Csr<VertexId, Value, SizeT> &graph_query,
    const Csr<VertexId, Value, SizeT> &graph_data,
    CommandLineArgs  &args,
    int          max_grid_size,
    int          num_gpus,
    double       max_queue_sizing,
    int          iterations,
    int          traversal_mode,
    CudaContext& context) {
    
    // define the problem data structure for graph primitive
    typedef SMProblem<VertexId, SizeT, Value> Problem;
	
    // INSTRUMENT specifies whether we want to keep such statistical data
    // allocate primitive enactor map
    SMEnactor<INSTRUMENT> sm_enactor(g_verbose);

    // allocate problem on the GPU
    // create a pointer of the SM Problem type
    Problem *sm_problem = new Problem;

    // allocate host-side array 
    VertexId *h_query_labels = (VertexId*)malloc(sizeof(VertexId) * graph_query.nodes);
    VertexId *h_data_labels = (VertexId*)malloc(sizeof(VertexId) * graph_data.nodes);
    VertexId *h_index = (VertexId*) malloc(sizeof(VertexId) * graph_query.nodes);
    VertexId *h_edge_index = (VertexId*)malloc(sizeof(VertexId) * graph_query.edges);
  
    bool *h_c_set = (bool*)malloc(sizeof(bool) * graph_data.nodes * graph_query.nodes);

    for (int i=0; i<graph_query.nodes; i++){
	h_index[i] = i;
    }
    for (int i=0; i<graph_query.edges; i++){
	h_edge_index[i] = i;
    }

    // copy data from CPU to GPU
    // initialize data members in DataSlice for graph
    util::GRError(sm_problem->Init(
      g_stream_from_host,
      graph_query,
      graph_data,
      h_query_labels,
      h_data_labels,
      h_index,
      h_edge_index,
      num_gpus),
      "Problem SM Initialization Failed", __FILE__, __LINE__);


    Stats *stats = new Stats("GPU Primitive");

    // perform calculation
    GpuTimer gpu_timer;

    float elapsed = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        util::GRError(
            sm_problem->Reset(sm_enactor.GetFrontierType(),
                               max_queue_sizing),
            "Problem Data Reset Failed", __FILE__, __LINE__);

        gpu_timer.Start();

	// launch sm enactor
        util::GRError(
            sm_enactor.template Enact<Problem>(context, sm_problem,
                max_grid_size, traversal_mode),
            "SM Problem Enact Failed", __FILE__, __LINE__);

        gpu_timer.Stop();

        elapsed += gpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    // extract results
    util::GRError(sm_problem->Extract(h_c_set),
        "SM Problem Data Extraction Failed", __FILE__, __LINE__);

    // compute reference CPU validation solution
    if (!g_quick) {
        printf("-- computing reference value ... (currently missing)\n");
        SimpleReference<VertexId, SizeT, Value>(graph_query, graph_data, h_c_set);
        printf("-- validation: (currently missing)\n");
    }

    // display solution
   // DisplaySolution<VertexId, SizeT, Value>(graph, h_c_set);

    // display statistics
    VertexId num_iteratios = 0;
    sm_enactor.GetStatistics(num_iteratios);
   // DisplayStats<VertexId, SizeT, Value>(*stats, graph, elapsed, num_iteratios);

    // clean up
    delete stats;
    if (sm_problem) delete sm_problem;
    if (h_c_set)    free(h_c_set);

    cudaDeviceSynchronize();
}

/**
 * @brief Test entry
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] context CudaContext pointer for ModernGPU APIs
 */
template<typename VertexId, typename SizeT, typename Value>
void RunTest(
    Csr<VertexId, Value, SizeT> &graph_query,
    Csr<VertexId, Value, SizeT> &graph_data,
    CommandLineArgs &args,
    CudaContext& context) {
    bool   instrumented     =   0;  // Collect instrumentation from kernels
    int    max_grid_size    =   0;  // Maximum grid size (0: up to the enactor)
    int    num_gpus         =   1;  // Number of GPUs for multi-GPU enactor
    double max_queue_sizing = 1.0;  // Maximum scaling factor for work queues
    int    iterations       =   1;  // Number of runs for testing
    int    traversal_mode   =  -1;  // Load-balanced or Dynamic cooperative
    g_quick                 =   0;  // Whether or not to skip CPU validation

    // choose traversal mode
    args.GetCmdLineArgument("traversal-mode", traversal_mode);
    if (traversal_mode == -1) {
        traversal_mode = graph_query.GetAverageDegree() > 8 ? 0 : 1;
    }

    g_verbose    = args.CheckCmdLineFlag("v");
    instrumented = args.CheckCmdLineFlag("instrumented");
    g_quick = args.CheckCmdLineFlag("quick");

    args.GetCmdLineArgument("iteration-num", iterations);
    args.GetCmdLineArgument("grid-size", max_grid_size);
    args.GetCmdLineArgument("queue-sizing", max_queue_sizing);

    if (instrumented) {
        RunTest<VertexId, Value, SizeT, true>(
            graph_query,
	    graph_data,
	    args,
            max_grid_size,
            num_gpus,
            max_queue_sizing,
            iterations,
            traversal_mode,
            context);
    } else {
        RunTest<VertexId, Value, SizeT, false>(
            graph_query,
	    graph_data,
	    args,
            max_grid_size,
            num_gpus,
            max_queue_sizing,
            iterations,
            traversal_mode,
            context);
    }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    CommandLineArgs args(argc, argv);
    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

    int device = 0;
    args.GetCmdLineArgument("device", device);
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // parse graph-construction parameters
    //g_undirected = args.CheckCmdLineFlag("undirected");
    g_undirected = true;

    std::string graph_type = argv[1];
    int flags = args.ParsedArgc();
    int graph_args = argc - flags - 1;
    if (graph_args < 1) {
        Usage();
        return 1;
    }

    typedef int VertexId;  // Use as the vertex identifier
    typedef int SizeT;     // Use as the graph size type
    typedef int Value;     // Use as the value type

    if (graph_type == "market") {
        // matrix-market coordinate-formatted graph
        Csr<VertexId, Value, SizeT> csr_query(false);
        Csr<VertexId, Value, SizeT> csr_data(false);

        char *market_filename = (graph_args == 2) ? argv[2] : NULL;

	// BuldMarketGraph() reads a mtx file into CSR data structure
	// Template argument = false because the grap has no edge weights
	
        if (graphio::BuildMarketGraph<true>(
            market_filename, csr_query, g_undirected, false) != 0) {
            return 1;
        }

        if (graphio::BuildMarketGraph<true>(
            market_filename, csr_data, g_undirected, false) != 0) {
            return 1;
        }

        csr_query.DisplayGraph();    // display graph adjacent list
        csr_data.DisplayGraph();    // display graph adjacent list
        csr_query.PrintHistogram();  // display graph histogram
        csr_data.PrintHistogram();  // display graph histogram

        RunTest(csr_query, csr_data, args, *context);  // run sm test

    } else {
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }
    return 0;
}
