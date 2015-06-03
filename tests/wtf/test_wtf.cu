// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_wtf.cu
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
#include <gunrock/app/wtf/wtf_enactor.cuh>
#include <gunrock/app/wtf/wtf_problem.cuh>
#include <gunrock/app/wtf/wtf_functor.cuh>

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
using namespace gunrock::app::wtf;


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

//bool g_verbose;
//bool g_undirected;
//bool g_quick;
//bool g_stream_from_host;

template <typename VertexId, typename Value>
struct RankPair {
    VertexId        vertex_id;
    Value           page_rank;

    RankPair(VertexId vertex_id, Value page_rank) : vertex_id(vertex_id), page_rank(page_rank) {}
};

template<typename RankPair>
bool PRCompare(
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
    printf("\ntest_wtf <graph type> <graph type args> [--device=<device_index>] "
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
 * @param[in] node_id Pointer to node ID array
 * @param[in] rank Pointer to node rank score array
 * @param[in] nodes Number of nodes in the graph.
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(VertexId *node_id, Value *rank, SizeT nodes)
{
    // Print out at most top 10 largest components
    int top = (nodes < 10) ? nodes : 10;
    printf("Top %d Page Ranks:\n", top);
    for (int i = 0; i < top; ++i)
    {
        printf("Vertex ID: %d, Page Rank: %5f\n", node_id[i], rank[i]);
    }
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
    Stats(const char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
    double                delta        ;// = 0.85f;        // Use whatever the specified graph-type's default is
    double                alpha        ;// = 0.2f;
    double                error        ;// = 0.01f;        // Error threshold
    long long             max_iter     ;// = 5;
    //bool                instrumented // = false;        // Whether or not to collect instrumentation from kernels
    //int                 max_grid_size       = 0;            // maximum grid size (0: leave it up to the enactor)
    //int                 num_gpus            = 1;            // Number of GPUs for multi-gpu enactor to use
    //VertexId            src                 = 0;            // Default source ID is 0
    //g_quick                                 = false;        // Whether or not to skip ref validation

    Test_Parameter()
    {   
        src       = 0;
        delta     = 0.85;
        alpha     = 0.2;
        error     = 0.01;
        max_iter  = 5;
    }   

    ~Test_Parameter()
    {   
    }   

    void Init(CommandLineArgs &args)
    {  
        TestParameter_Base::Init(args);
        args.GetCmdLineArgument("delta", delta);
        args.GetCmdLineArgument("alpha", alpha);
        args.GetCmdLineArgument("error", error);
        args.GetCmdLineArgument("max-iter", max_iter);
    }   
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] h_rank Host-side vector stores computed page rank values for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] total_queued Total element queued in WTF kernel running process
 * @param[in] avg_duty Average duty of the WTF kernels
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void DisplayStats(
    Stats               &stats,
    Value               *h_rank,
    const Csr<VertexId, Value, SizeT> &graph,
    double              elapsed,
    long long           total_queued,
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

/******************************************************************************
 * WTF Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference WTF implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] src Source node ID for WTF algorithm
 * @param[out] node_id Pointer to store computed output node ID
 * @param[in] rank Host-side vector to store CPU computed labels for each node
 * @param[in] delta Delta value for computing PageRank score
 * @param[in] alpha Parameter to adjust iteration number
 * @param[in] max_iter max iteration to go
 */
// TODO: Boost PageRank cannot handle personalized pagerank, so currently the CPU
// implementation gives incorrect answer. Need to find a CPU PPR implementation
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void SimpleReferenceHITS(
    const Csr<VertexId, Value, SizeT>       &graph,
    VertexId                                src,
    VertexId                                *node_id,
    Value                                   *rank,
    Value                                   delta,
    Value                                   alpha,
    SizeT                                   max_iter)
{
    using namespace boost;

    //Preparation
    typedef adjacency_list<vecS, vecS, bidirectionalS, no_property,
                           property<edge_index_t, int> > Graph;

    Graph g;

    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
        {
            Graph::edge_descriptor e =
                add_edge(i, graph.column_indices[j], g).first;
            put(edge_index, g, e, i);
        }
    }


    //
    //compute page rank
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    //remove_dangling_links(g);

    std::vector<Value> ranks(num_vertices(g));
    page_rank(g, make_iterator_property_map(
                  ranks.begin(), get(boost::vertex_index, g)),
              boost::graph::n_iterations(max_iter));

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    for (std::size_t i = 0; i < num_vertices(g); ++i)
    {
        rank[i] = ranks[i];
    }

    //sort the top page ranks
    RankPair<SizeT, Value> *pr_list =
        (RankPair<SizeT, Value>*)malloc(
            sizeof(RankPair<SizeT, Value>) * num_vertices(g));
    for (int i = 0; i < num_vertices(g); ++i)
    {
        pr_list[i].vertex_id = i;
        pr_list[i].page_rank = rank[i];
    }
    std::stable_sort(
        pr_list, pr_list + num_vertices(g), PRCompare<RankPair<SizeT, Value> >);

    std::vector<int> in_degree(num_vertices(g));
    std::vector<Value> refscore(num_vertices(g));

    for (int i = 0; i < num_vertices(g); ++i)
    {
        node_id[i] = pr_list[i].vertex_id;
        rank[i] = (i == src) ? 1.0 : 0;
        in_degree[i] = 0;
        refscore[i] = 0;
    }

    free(pr_list);

    int cot_size = (graph.nodes > 1000) ? 1000 : graph.nodes;

    for (int i = 0; i < cot_size; ++i)
    {
        int node = node_id[i];
        for (int j = graph.row_offsets[node];
             j < graph.row_offsets[node+1]; ++j)
        {
            VertexId edge = graph.column_indices[j];
            ++in_degree[edge];
        }
    }

    int salsa_iter = 1.0/alpha+1;
    for (int iter = 0; iter < salsa_iter; ++iter)
    {
        for (int i = 0; i < cot_size; ++i)
        {
            int node = node_id[i];
            int out_degree = graph.row_offsets[node+1]-graph.row_offsets[node];
            for (int j = graph.row_offsets[node];
                 j < graph.row_offsets[node+1]; ++j)
            {
                VertexId edge = graph.column_indices[j];
                Value val = rank[node]/ (out_degree > 0 ? out_degree : 1.0);
                refscore[edge] += val;
            }
        }
        for (int i = 0; i < cot_size; ++i)
        {
            rank[node_id[i]] = 0;
        }

        for (int i = 0; i < cot_size; ++i)
        {
            int node = node_id[i];
            rank[node] += (node == src) ? alpha : 0;
            for (int j = graph.row_offsets[node];
                 j < graph.row_offsets[node+1]; ++j)
            {
                VertexId edge = graph.column_indices[j];
                Value val = (1-alpha)*refscore[edge]/in_degree[edge];
                rank[node] += val;
            }
        }

        for (int i = 0; i < cot_size; ++i)
        {
            if (iter+1<salsa_iter) refscore[node_id[i]] = 0;
        }
    }

    //sort the top page ranks
    RankPair<SizeT, Value> *final_list =
        (RankPair<SizeT, Value>*)malloc(
            sizeof(RankPair<SizeT, Value>) * num_vertices(g));
    for (int i = 0; i < num_vertices(g); ++i)
    {
        final_list[i].vertex_id = node_id[i];
        final_list[i].page_rank = refscore[i];
    }
    std::stable_sort(
        final_list, final_list + num_vertices(g),
        PRCompare<RankPair<SizeT, Value> >);

    for (int i = 0; i < num_vertices(g); ++i)
    {
        node_id[i] = final_list[i].vertex_id;
        rank[i] = final_list[i].page_rank;
    }

    free(final_list);

    printf("CPU Who-To-Follow finished in %lf msec.\n", elapsed);
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
 * @param[in] src Source node ID for WTF algorithm
 * @param[in] delta Delta value for computing WTF, usually set to .85
 * @param[in] alpha Parameter to adjust iteration number
 * @param[in] error Error threshold value
 * @param[in] max_iter Max iteration for WTF computing
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
    VertexId src,
    Value delta,
    Value alpha,
    Value error,
    SizeT max_iter,
    int max_grid_size,
    int num_gpus,
    CudaContext& context)*/
{

    typedef WTFProblem<
        VertexId,
        SizeT,
        Value> Problem;

    Csr<VertexId, Value, SizeT>
                 *graph                 = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    VertexId      src                   = (VertexId)parameter -> src;
    int           max_grid_size         = parameter -> max_grid_size;
    int           num_gpus              = parameter -> num_gpus;
    //double        max_queue_sizing      = parameter -> max_queue_sizing;
    ContextPtr   *context               = (ContextPtr*)parameter -> context;
    int          *gpu_idx               = parameter -> gpu_idx;
    //cudaStream_t *streams               = parameter -> streams;
    bool          g_quick               = parameter -> g_quick;
    bool          g_stream_from_host    = parameter -> g_stream_from_host;
    //bool          g_undirected          = parameter -> g_undirected;
    Value         alpha                 = parameter -> alpha;
    Value         delta                 = parameter -> delta;
    Value         error                 = parameter -> error;
    SizeT         max_iter              = parameter -> max_iter;
    // Allocate host-side label array (for both reference and gpu-computed results)
    Value    *reference_rank    = (Value*)malloc(sizeof(Value) * graph->nodes);
    Value    *h_rank            = (Value*)malloc(sizeof(Value) * graph->nodes);
    VertexId *h_node_id         = (VertexId*)malloc(sizeof(VertexId) * graph->nodes);
    VertexId *reference_node_id = (VertexId*)malloc(sizeof(VertexId) * graph->nodes);
    Value    *reference_check   = (g_quick) ? NULL : reference_rank;

    // Allocate WTF enactor map
    WTFEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> wtf_enactor(gpu_idx);
    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      g_stream_from_host,
                      *graph,
                      num_gpus),
                  "Problem WTF Initialization Failed", __FILE__, __LINE__);

    Stats *stats = new Stats("GPU Who-To-Follow");

    long long           total_queued = 0;
    double              avg_duty = 0.0;

    // Perform WTF
    GpuTimer gpu_timer;

    util::GRError(
        csr_problem->Reset(
            src, delta, alpha, error, wtf_enactor.GetFrontierType()),
        "pr Problem Data Reset Failed", __FILE__, __LINE__);
    gpu_timer.Start();
    util::GRError(
        wtf_enactor.template Enact<Problem>(
            *context, src, alpha, csr_problem, max_iter, max_grid_size),
        "HITS Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();

    wtf_enactor.GetStatistics(total_queued, avg_duty);

    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        csr_problem->Extract(h_rank, h_node_id),
        "HITS Problem Data Extraction Failed", __FILE__, __LINE__);

    float total_pr = 0;
    for (int i = 0; i < graph->nodes; ++i)
    {
        total_pr += h_rank[i];
    }

    //
    // Compute reference CPU HITS solution for source-distance
    //
    if (reference_check != NULL && total_pr > 0)
    {
        printf("compute ref value\n");
        SimpleReferenceHITS(
            *graph,
            src,
            reference_node_id,
            reference_check,
            delta,
            alpha,
            max_iter);
        printf("\n");
    }

    // Verify the result
    if (reference_check != NULL && total_pr > 0)
    {
        printf("Validity: ");
        CompareResults(h_rank, reference_check, graph->nodes, true);
    }
    printf("\nGPU result.");
    // Display Solution
    DisplaySolution(h_node_id, h_rank, graph->nodes);

    DisplayStats(
        *stats,
        h_rank,
        *graph,
        elapsed,
        total_queued,
        avg_duty);


    // Cleanup
    delete stats;
    if (csr_problem) delete csr_problem;
    if (reference_check) free(reference_check);
    if (h_rank) free(h_rank);

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

template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> *graph,
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
    bool g_undirected = args.CheckCmdLineFlag("undirected");

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
        //csr.DisplayGraph();

        // Run tests
        RunTests(&csr, args, 1, &context, &dev);
    }
    else
    {
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }
    return 0;
}

