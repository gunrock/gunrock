// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_hits.cu
 *
 * @brief Simple test driver program for using HITS algorithm to compute rank.
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
#include <gunrock/app/hits/hits_enactor.cuh>
#include <gunrock/app/hits/hits_problem.cuh>
#include <gunrock/app/hits/hits_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::hits;


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

template <typename VertexId, typename Value>
struct RankPair
{
    VertexId        vertex_id;
    Value           page_rank;

    RankPair(VertexId vertex_id, Value page_rank) :
        vertex_id(vertex_id), page_rank(page_rank) {}
};

template<typename RankPair>
bool HITSCompare(
    RankPair elem1,
    RankPair elem2)
{
    return elem1.page_rank > elem2.page_rank;
}

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf("\ntest_hits <graph type> <graph type args> [--device=<device_index>] "
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
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] hrank Pointer to hub rank score array
 * @param[in] arank Pointer to authority rank score array
 * @param[in] nodes Number of nodes in the graph.
 */
template<typename Value, typename SizeT>
void DisplaySolution(Value *hrank, Value *arank, SizeT nodes)
{
    //sort the top page ranks
    RankPair<SizeT, Value> *hr_list =
        (RankPair<SizeT, Value>*)malloc(sizeof(RankPair<SizeT, Value>) * nodes);
    RankPair<SizeT, Value> *ar_list =
        (RankPair<SizeT, Value>*)malloc(sizeof(RankPair<SizeT, Value>) * nodes);

    for (int i = 0; i < nodes; ++i)
    {
        hr_list[i].vertex_id = i;
        hr_list[i].page_rank = hrank[i];
        ar_list[i].vertex_id = i;
        ar_list[i].page_rank = arank[i];
    }
    std::stable_sort(
        hr_list, hr_list + nodes, HITSCompare<RankPair<SizeT, Value> >);
    std::stable_sort(
        ar_list, ar_list + nodes, HITSCompare<RankPair<SizeT, Value> >);

    // Print out at most top 10 largest components
    int top = (nodes < 10) ? nodes : 10;
    printf("Top %d Ranks:\n", top);
    for (int i = 0; i < top; ++i)
    {
        printf("Vertex ID: %d, Hub Rank: %5f\n",
               hr_list[i].vertex_id, hr_list[i].page_rank);
        printf("Vertex ID: %d, Authority Rank: %5f\n",
               ar_list[i].vertex_id, ar_list[i].page_rank);
    }

    free(hr_list);
    free(ar_list);
}

/**
 * Performance/Evaluation statistics
 */

struct Stats {
    const char *name;
    Statistic rate;
    Statistic search_depth;
    Statistic redundant_work;
    Statistic duty;

    Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
    Stats(const char *name) :
        name(name), rate(), search_depth(), redundant_work(), duty() {}
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] avg_duty Average duty of the BFS kernels
 */

void DisplayStats(
    Stats               &stats,
    double              elapsed,
    double              avg_duty)
{

    // Display test name
    printf("[%s] finished. ", stats.name);

    // Display the specific sample statistics
    printf(" elapsed: %.3f ms", elapsed);
    if (avg_duty != 0) {
        printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
    }
    printf("\n");
}

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
    double    delta        ;// = 0.2;
    long long src          ;// = 0;
    long long max_iter     ;// = 1;
    void*     inv_graph    ;

    Test_Parameter()
    {
        delta = 0.2;
        src   = 0;
        max_iter = 1;
        inv_graph = NULL;
    }

    ~Test_Parameter()
    {
    }

    void Init(CommandLineArgs &args)
    {
        TestParameter_Base::Init(args);
        args.GetCmdLineArgument("delta", delta);
        args.GetCmdLineArgument("max-iter", max_iter);
    }
};


/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference HITS implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] inv_graph Reference to the inversed CSR graph we process on
 * @param[in] hrank Host-side vector to store CPU computed hub rank scores for each node
 * @param[in] arank Host-side vector to store CPU computed authority rank scores for each node
 * @param[in] max_iter max iteration to go
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void SimpleReferenceHITS(
    const Csr<VertexId, Value, SizeT>       &graph,
    const Csr<VertexId, Value, SizeT>       &inv_graph,
    Value                                   *hrank,
    Value                                   *arank,
    SizeT                                   max_iter)
{
    //using namespace boost;

    //Preparation

    //
    //compute HITS rank
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printf("CPU BFS finished in %lf msec.\n", elapsed);
}

/**
 * @brief Run HITS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] inv_graph Reference to the inversed CSR graph we process on
 * @param[in] src Source node ID for HITS algorithm
 * @param[in] delta Delta value for computing HITS, usually set to .85
 * @param[in] max_iter Max iteration for HITS computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 *
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK>
void RunTests(Test_Parameter *parameter)
    /*const Csr<VertexId, Value, SizeT> &graph,
    const Csr<VertexId, Value, SizeT> &inv_graph,
    VertexId src,
    Value delta,
    SizeT max_iter,
    int max_grid_size,
    int num_gpus,
    CudaContext& context)*/
{

    typedef HITSProblem<
        VertexId,
        SizeT,
        Value> Problem;

    Csr<VertexId, Value, SizeT>
                 *graph                 = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    Csr<VertexId, Value, SizeT>
                 *inv_graph             = (Csr<VertexId, Value, SizeT>*)parameter->inv_graph;
    VertexId      src                   = (VertexId)parameter -> src;
    int           max_grid_size         = parameter -> max_grid_size;
    SizeT         max_iter              = parameter -> max_iter;
    Value         delta                 = parameter -> delta;
    int           num_gpus              = parameter -> num_gpus;
    ContextPtr   *context               = (ContextPtr*)parameter -> context;
    int          *gpu_idx               = parameter -> gpu_idx;
    bool          g_quick               = parameter -> g_quick;
    bool          g_stream_from_host    = parameter -> g_stream_from_host;
    // Allocate host-side label array (for both reference and gpu-computed results)
    Value    *reference_hrank       = (Value*)malloc(sizeof(Value) * graph->nodes);
    Value    *reference_arank       = (Value*)malloc(sizeof(Value) * graph->nodes);
    Value    *h_hrank               = (Value*)malloc(sizeof(Value) * graph->nodes);
    Value    *h_arank               = (Value*)malloc(sizeof(Value) * graph->nodes);
    Value    *reference_check_h     = (g_quick) ? NULL : reference_hrank;
    Value    *reference_check_a     = (g_quick) ? NULL : reference_arank;

    // Allocate BFS enactor map
    HITSEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> hits_enactor(gpu_idx);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      g_stream_from_host,
                      *graph,
                      *inv_graph,
                      num_gpus), "Problem HITS Initialization Failed", __FILE__, __LINE__);

    //
    // Compute reference CPU HITS solution for source-distance
    //
    if (reference_check_h != NULL)
    {
        printf("compute ref value\n");
        SimpleReferenceHITS(
            *graph,
            *inv_graph,
            reference_check_h,
            reference_check_a,
            max_iter);
        printf("\n");
    }

    Stats *stats = new Stats("GPU HITS");

    long long           total_queued = 0;
    double              avg_duty = 0.0;

    // Perform HITS
    GpuTimer gpu_timer;

    util::GRError(
        csr_problem->Reset(src, delta, hits_enactor.GetFrontierType()),
        "HITS Problem Data Reset Failed", __FILE__, __LINE__);
    gpu_timer.Start();
    util::GRError(
        hits_enactor.template Enact<Problem>(
            *context, csr_problem, max_iter, max_grid_size),
        "HITS Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();

    hits_enactor.GetStatistics(total_queued, avg_duty);

    double elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        csr_problem->Extract(h_hrank, h_arank),
        "HITS Problem Data Extraction Failed", __FILE__, __LINE__);

    // Verify the result
    if (reference_check_a != NULL)
    {
        printf("Validity: ");
        CompareResults(h_hrank, reference_check_h, graph->nodes, true);
        CompareResults(h_arank, reference_check_a, graph->nodes, true);
    }

    printf("\nFirst 40 labels of the GPU result.");
    // Display Solution
    DisplaySolution(h_hrank, h_arank, graph->nodes);

    DisplayStats(
        *stats,
        elapsed,
        avg_duty);

    // Cleanup
    delete stats;
    if (csr_problem) delete csr_problem;
    if (reference_check_h) free(reference_check_h);
    if (reference_check_a) free(reference_check_a);

    if (h_hrank) free(h_hrank);
    if (h_arank) free(h_arank);

    cudaDeviceSynchronize();
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG>
void RunTests_size_check(Test_Parameter *parameter)
{
    if (parameter->size_check) RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        true > (parameter);
   else RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        false> (parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT>
void RunTests_debug(Test_Parameter *parameter)
{
    if (parameter->debug) RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        true > (parameter);
    else RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        false> (parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT>
void RunTests_instrumented(Test_Parameter *parameter)
{
    if (parameter->instrumented) RunTests_debug
        <VertexId, Value, SizeT,
        true > (parameter);
    else RunTests_debug
        <VertexId, Value, SizeT,
        false> (parameter);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] inv_graph Reference to the inversed CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] context CudaContext for moderngpu to use
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> *graph,
    Csr<VertexId, Value, SizeT> *inv_graph,
    CommandLineArgs             &args,
    int                          num_gpus,
    ContextPtr                  *context,
    int                         *gpu_idx,
    cudaStream_t                *streams = NULL)
{
    string src_str="";
    Test_Parameter *parameter = new Test_Parameter;

    parameter -> Init(args);
    parameter -> graph              = graph;
    parameter -> inv_graph          = inv_graph;
    parameter -> num_gpus           = num_gpus;
    parameter -> context            = context;
    parameter -> gpu_idx            = gpu_idx;
    parameter -> streams            = streams;

    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty()) {
        parameter->src = 0;
    } else if (src_str.compare("randomize") == 0) {
        parameter->src = graphio::RandomNode(graph->nodes);
    } else if (src_str.compare("largestdegree") == 0) {
        int temp;
        parameter->src = graph->GetNodeWithHighestDegree(temp);
    } else {
        args.GetCmdLineArgument("src", parameter->src);
    }

    RunTests_instrumented<VertexId, Value, SizeT>(parameter);
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
    bool g_undirected = false;

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
        typedef float Value;                    // Use as the value type
        typedef int SizeT;                      // Use as the graph size type
        Csr<VertexId, Value, SizeT> csr(false); // default for stream_from_host

        Csr<VertexId, Value, SizeT> inv_csr(false);

        if (graph_args < 1) { Usage(); return 1; }

        char *market_filename = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
                market_filename,
                csr,
                g_undirected,
                false) != 0)
        {
            return 1;
        }

        if (graphio::BuildMarketGraph<false>(
                market_filename,
                inv_csr,
                g_undirected,
                true) != 0)
        {
            return 1;
        }

        csr.PrintHistogram();
        //csr.DisplayGraph();
        //inv_csr.DisplayGraph();

        // Run tests
        RunTests(&csr, &inv_csr, args, 1, &context, &dev);

    }
    else
    {
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }
    return 0;
}
 
