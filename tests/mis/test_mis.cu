// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_mis.cu
 *
 * @brief Simple test driver program for computing Pagerank.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <cstdlib>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/mis/mis_enactor.cuh>
#include <gunrock/app/mis/mis_problem.cuh>
#include <gunrock/app/mis/mis_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

// boost includes
#include <boost/config.hpp>
#include <boost/utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/page_rank.hpp>


using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::mis;


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
    printf("\ntest_mis <graph type> <graph type args> [--device=<device_index>] "
           "[--undirected] [--instrumented] [--quick] "
           "[--v]\n"
           "\n"
           "Graph types and args:\n"
           "  market [<file>]\n"
           "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
           "    edges from stdin (or from the optionally-specified file).\n"
           "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
           "  --undirected If set then treat the graph as undirected.\n"
           "  --instrumented If set then kernels keep track of queue-search_depth\n"
           "  and barrier duty (a relative indicator of load imbalance.)\n"
           "  --quick If set will skip the CPU validation code.\n"
        );
}

/**
 * @brief Displays the MIS ID result (i.e., graph coloring ID)
 *
 * @param[in] mis_ids maximal independent ids
 * @param[in] nodes number of nodes in graph
 */
template<typename Value, typename SizeT>
void DisplaySolution( Value *mis_ids, SizeT nodes)
{
    if (nodes > 40)
        nodes = 40;
    printf("[");
    for (SizeT i = 0; i < nodes; ++i) {
        PrintValue(i);
        printf(":");
        PrintValue(mis_ids[i]);
        printf(" ");
    }
    printf("]\n");
}

/**
 * Performance/Evaluation statistics
 */
struct Stats
{
    const char *name;
    Statistic rate;
    Statistic search_depth;
    Statistic redundant_work;
    Statistic duty;

    Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
    Stats(const char *name) :
        name(name), rate(), search_depth(), redundant_work(), duty() {}
};

/******************************************************************************
 * MIS Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference MIS implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] misid Host-side vector to store CPU computed mid ids for each node
 * @param[in] max_iter max iteration to go
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void SimpleReferenceMis(
    const Csr<VertexId, Value, SizeT>       &graph,
    Value                                   *misid,
    SizeT                                   max_iter)
{
    printf("CPU MIS missing now.\n");
}

/**
 * @brief Run MIS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] max_iter Max iteration for Page Rank computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] iterations Number of iterations for running the test
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
    SizeT max_iter,
    int max_grid_size,
    int num_gpus,
    int iterations,
    CudaContext& context)
{

    typedef MISProblem<
        VertexId,
        SizeT,
        Value> Problem;

    // Allocate host-side label array (for both reference and gpu-computed results)
    Value    *reference_misid    = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value    *h_misid            = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value    *reference_check   = (g_quick) ? NULL : reference_misid;

    // Allocate MIS enactor map
    MISEnactor<INSTRUMENT> mis_enactor(g_verbose);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      g_stream_from_host,
                      graph,
                      num_gpus),
                  "Problem mis Initialization Failed", __FILE__, __LINE__);

    Stats *stats = new Stats("GPU MIS");

    long long           total_queued = 0;
    double              avg_duty = 0.0;

    // Perform MIS
    GpuTimer gpu_timer;

    float elapsed = 0.0f;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(
            csr_problem->Reset(mis_enactor.GetFrontierType()),
            "MIS Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(
            mis_enactor.template Enact<Problem>(
                context, csr_problem, max_iter, max_grid_size),
            "MIS Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();
        elapsed += gpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    mis_enactor.GetStatistics(total_queued, avg_duty);

    // Copy out results
    util::GRError(
        csr_problem->Extract(h_misid),
        "MIS Problem Data Extraction Failed", __FILE__, __LINE__);

    //
    // Compute reference CPU MIS solution for source-distance
    //
    if (reference_check != NULL)
    {
        printf("Computing reference value ...\n");
        SimpleReferenceMis(
            graph,
            reference_check,
            max_iter);
        printf("\n");
    }

    // Verify the result
    if (reference_check != NULL)
    {
        printf("Validity: ");
        CompareResults(h_misid, reference_check, graph.nodes, true);
    }

DisplaySolution(h_misid, graph.nodes);
    printf("\nFirst 40 labels of the GPU result.\n");


    // Cleanup
    delete stats;
    if (csr_problem) delete csr_problem;
    if (reference_check) free(reference_check);
    if (h_misid) free(h_misid);

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
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    CommandLineArgs &args,
    CudaContext& context)
{
    SizeT    max_iter      = 20;
    bool     instrumented  = false; // Whether or not to collect instrumentation from kernels
    int      max_grid_size = 0;     // maximum grid size (0: leave it up to the enactor)
    int      num_gpus      = 1;     // Number of GPUs for multi-gpu enactor to use
    int      iterations    = 1;

    instrumented = args.CheckCmdLineFlag("instrumented");
    args.GetCmdLineArgument("max-iter", max_iter);
    args.GetCmdLineArgument("iteration-num", iterations);
    g_quick = args.CheckCmdLineFlag("quick");
    g_verbose = args.CheckCmdLineFlag("v");

    if (instrumented)
    {
        RunTests<VertexId, Value, SizeT, true>(
            graph,
            max_iter,
            max_grid_size,
            num_gpus,
            iterations,
            context);
    }
    else
    {
        RunTests<VertexId, Value, SizeT, false>(
            graph,
            max_iter,
            max_grid_size,
            num_gpus,
            iterations,
            context);
    }
}



/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)
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
    ContextPtr context = mgpu::CreateCudaDevice(dev);

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

        typedef int VertexId;                   // Use as the node identifier
        typedef int Value;                    // Use as the value type
        typedef int SizeT;                      // Use as the graph size type
        Csr<VertexId, Value, SizeT> csr(false); // default for stream_from_host

        if (graph_args < 1) { Usage(); return 1; }
        char *market_filename = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
                market_filename,
                csr,
                g_undirected,
                false) != 0) // no inverse graph
        {
            return 1;
        }

        csr.PrintHistogram();

        // Run tests
        RunTests(csr, args, *context);

    }
    else
    {
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }
    return 0;
}
