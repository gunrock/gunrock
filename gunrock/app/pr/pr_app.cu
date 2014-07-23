// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cu
 *
 * @brief Gunrock Computing Pagerank Implementation
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/pr/pr_enactor.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/app/pr/pr_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::pr;


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/
static bool g_verbose;
//static bool g_undirected;
//static bool g_quick;
static bool g_stream_from_host;

template <typename VertexId, typename Value>
struct RankPair {
    VertexId vertex_id;
    Value    page_rank;
    RankPair(VertexId vertex_id, Value page_rank) : vertex_id(vertex_id), page_rank(page_rank) {}
};

template<typename RankPair>
__inline__ bool PRCompare(
    RankPair elem1,
    RankPair elem2)
{
    return elem1.page_rank > elem2.page_rank;
}

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
/*
static void Usage()
{
    printf(
        "\ntest_pr <graph type> <graph type args> [--device=<device_index>] "
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
*/

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] nodes Number of nodes in the graph.
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void DisplaySolution(VertexId *node_id, Value *rank, SizeT nodes)
{
    printf("\nFirst %d labels of the GPU result.", nodes);
    // Print out at most top 10 largest
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
 * @param[in] total_queued Total element queued in BFS kernel running process
 * @param[in] avg_duty Average duty of the BFS kernels
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
    if (avg_duty != 0)
    {
        printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
    }
    printf("\n");
}

/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/
/**
 * @brief Run PR tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] delta Delta value for computing PageRank, usually set to .85
 * @param[in] error Error threshold value
 * @param[in] max_iter Max iteration for Page Rank computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 *
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void run_page_rank(
    GunrockGraph *ggraph_out,
    VertexId     *node_ids,
    Value        *page_rank,
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId     source,
    Value        delta,
    Value        error,
    SizeT        max_iter,
    int          max_grid_size,
    int          num_gpus,
    CudaContext& context)
{
    typedef PRProblem<
        VertexId,
        SizeT,
        Value> Problem;

    // Allocate host-side label array for gpu-computed results
    //Value    *h_rank    = (Value*)malloc(sizeof(Value) * graph.nodes);
    //VertexId *h_node_id = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);

    // Allocate BFS enactor map
    PREnactor<false> pr_enactor(g_verbose);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
        g_stream_from_host,
        graph,
        num_gpus),
    "Problem PR Initialization Failed", __FILE__, __LINE__);

    Stats *stats = new Stats("GPU PageRank");

    long long total_queued = 0;
    double    avg_duty = 0.0;

    // Perform BFS
    GpuTimer gpu_timer;

    util::GRError(csr_problem->Reset(
        source, delta, error, pr_enactor.GetFrontierType()),
        "PR Problem Data Reset Failed", __FILE__, __LINE__);
    gpu_timer.Start();
    util::GRError(pr_enactor.template Enact<Problem>(
        context, csr_problem, max_iter, max_grid_size),
        "PR Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();

    pr_enactor.GetStatistics(total_queued, avg_duty);
    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(csr_problem->Extract(page_rank, node_ids),
        "PageRank Problem Data Extraction Failed", __FILE__, __LINE__);

    // Display Solution
    //DisplaySolution(node_ids, page_rank, graph.nodes);

    DisplayStats(
        *stats,
        page_rank,
        graph,
        elapsed,
        total_queued,
        avg_duty);

    // Cleanup
    delete stats;
    if (csr_problem) delete csr_problem;
    //if (h_rank) free(h_rank);

    cudaDeviceSynchronize();
}

/**
 * @brief run_page_rank entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 */
void gunrock_pr(
    GunrockGraph       *ggraph_out,
    void               *node_ids,
    void               *page_rank,
    const GunrockGraph *ggraph_in,
    GunrockConfig      pr_config,
    GunrockDataType    data_type)
{
    // moderngpu preparations
    int device = 0;
    device = pr_config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // build input csr format graph
    Csr<int, float, int> csr_graph(false);
    csr_graph.nodes = ggraph_in->num_nodes;
    csr_graph.edges = ggraph_in->num_edges;
    csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
    csr_graph.column_indices = (int*)ggraph_in->col_indices;

    // page rank configurations
    float delta         = 0.85f; //!< use whatever the specified graph-type's default is
    float error         = 0.01f; //!< error threshold
    int   max_iter      = 20;    //!< maximum number of iterations
    int   max_grid_size = 0;     //!< maximum grid size (0: leave it up to the enactor)
    int   num_gpus      = 1;     //!< number of GPUs for multi-gpu enactor to use
    int   source        = -1;    //!< source node to start

    delta    = pr_config.delta;
    error    = pr_config.error;
    source   = pr_config.source;
    max_iter = pr_config.max_iter;

    run_page_rank<int, float, int>(
        ggraph_out,
        (int*)node_ids,
        (float*)page_rank,
        csr_graph,
        source,
        delta,
        error,
        max_iter,
        max_grid_size,
        num_gpus,
        *context);

    // reset for free memory
    csr_graph.row_offsets    = NULL;
    csr_graph.column_indices = NULL;
    csr_graph.row_offsets    = NULL;
    csr_graph.column_indices = NULL;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
