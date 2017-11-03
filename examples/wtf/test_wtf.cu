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
using namespace gunrock::app;
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
struct RankPair
{
    VertexId        vertex_id;
    Value           page_rank;

    RankPair(VertexId vertex_id, Value page_rank) : 
        vertex_id(vertex_id), page_rank(page_rank) {}
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
    printf(
        "test <graph-type> [graph-type-arguments]\n"
        "Graph type and graph type arguments:\n"
        "    market <matrix-market-file-name>\n"
        "        Reads a Matrix-Market coordinate-formatted graph of\n"
        "        directed/undirected edges from STDIN (or from the\n"
        "        optionally-specified file).\n"
        "    rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)\n"
        "        Generate R-MAT graph as input\n"
        "        --rmat_scale=<vertex-scale>\n"
        "        --rmat_nodes=<number-nodes>\n"
        "        --rmat_edgefactor=<edge-factor>\n"
        "        --rmat_edges=<number-edges>\n"
        "        --rmat_a=<factor> --rmat_b=<factor> --rmat_c=<factor>\n"
        "        --rmat_seed=<seed>\n"
        "    rgg (default: rgg_scale = 10, rgg_thfactor = 0.55)\n"
        "        Generate Random Geometry Graph as input\n"
        "        --rgg_scale=<vertex-scale>\n"
        "        --rgg_nodes=<number-nodes>\n"
        "        --rgg_thfactor=<threshold-factor>\n"
        "        --rgg_threshold=<threshold>\n"
        "        --rgg_vmultipiler=<vmultipiler>\n"
        "        --rgg_seed=<seed>\n\n"
        "Optional arguments:\n"
        "[--device=<device_index>] Set GPU(s) for testing (Default: 0).\n"
        "[--undirected]            Treat the graph as undirected (symmetric).\n"
        "[--instrumented]          Keep kernels statics [Default: Disable].\n"
        "                          total_queued, search_depth and barrier duty.\n"
        "                          (a relative indicator of load imbalance.)\n"
        "[--quick]                 Skip the CPU reference validation process.\n"
        "[--mark-pred]             Keep both label info and predecessor info.\n"
        "[--disable-size-check]    Disable frontier queue size check.\n"
        "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
        "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--in-sizing=<in/out_queue_scale_factor>]\n"
        "                          Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] node_id Pointer to node ID array
 * @param[in] rank Pointer to node rank score array
 * @param[in] nodes Number of nodes in the graph.
 */
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(VertexId *node_id, Value *rank, SizeT nodes)
{
    // Print out at most top 10 largest components
    SizeT top = (nodes < 10) ? nodes : 10;
    printf("Top %lld Page Ranks:\n", (long long)top);
    for (SizeT i = 0; i < top; ++i)
    {
        printf("Vertex ID: %lld, Page Rank: %5f\n", 
            (long long)node_id[i], rank[i]);
    }
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
template <
    typename VertexId,
    typename SizeT,
    typename Value>
void ReferenceWTF(
    const Csr<VertexId, SizeT, Value>       &graph,
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
        for (SizeT j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
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
    for (std::size_t i = 0; i < num_vertices(g); ++i)
    {
        pr_list[i].vertex_id = i;
        pr_list[i].page_rank = rank[i];
    }
    std::stable_sort(
        pr_list, pr_list + num_vertices(g), PRCompare<RankPair<SizeT, Value> >);

    std::vector<SizeT> in_degree(num_vertices(g));
    std::vector<Value> refscore(num_vertices(g));

    for (std::size_t i = 0; i < num_vertices(g); ++i)
    {
        node_id[i] = pr_list[i].vertex_id;
        rank[i] = (i == src) ? 1.0 : 0;
        in_degree[i] = 0;
        refscore[i] = 0;
    }

    free(pr_list);

    SizeT cot_size = (graph.nodes > 1000) ? 1000 : graph.nodes;

    for (SizeT i = 0; i < cot_size; ++i)
    {
        VertexId node = node_id[i];
        for (SizeT j = graph.row_offsets[node];
                j < graph.row_offsets[node + 1]; ++j)
        {
            VertexId edge = graph.column_indices[j];
            ++in_degree[edge];
        }
    }

    SizeT salsa_iter = 1.0 / alpha + 1;
    for (SizeT iter = 0; iter < salsa_iter; ++iter)
    {
        for (SizeT i = 0; i < cot_size; ++i)
        {
            VertexId node = node_id[i];
            SizeT out_degree = graph.row_offsets[node + 1] - graph.row_offsets[node];
            for (SizeT j = graph.row_offsets[node];
                    j < graph.row_offsets[node + 1]; ++j)
            {
                VertexId edge = graph.column_indices[j];
                Value val = rank[node] / (out_degree > 0 ? out_degree : 1.0);
                refscore[edge] += val;
            }
        }
        for (SizeT i = 0; i < cot_size; ++i)
        {
            rank[node_id[i]] = 0;
        }

        for (SizeT i = 0; i < cot_size; ++i)
        {
            VertexId node = node_id[i];
            rank[node] += (node == src) ? alpha : 0;
            for (SizeT j = graph.row_offsets[node];
                    j < graph.row_offsets[node + 1]; ++j)
            {
                VertexId edge = graph.column_indices[j];
                Value val = (1 - alpha) * refscore[edge] / in_degree[edge];
                rank[node] += val;
            }
        }

        for (SizeT i = 0; i < cot_size; ++i)
        {
            if (iter + 1 < salsa_iter) refscore[node_id[i]] = 0;
        }
    }

    //sort the top page ranks
    RankPair<SizeT, Value> *final_list =
        (RankPair<SizeT, Value>*)malloc(
            sizeof(RankPair<SizeT, Value>) * num_vertices(g));
    for (std::size_t i = 0; i < num_vertices(g); ++i)
    {
        final_list[i].vertex_id = node_id[i];
        final_list[i].page_rank = refscore[i];
    }
    std::stable_sort(
        final_list, final_list + num_vertices(g),
        PRCompare<RankPair<SizeT, Value> >);

    for (std::size_t i = 0; i < num_vertices(g); ++i)
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
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK >
void RunTests(Info<VertexId, SizeT, Value> *info)
{
    typedef WTFProblem <
        VertexId,
        SizeT,
        Value > 
        Problem;
    typedef WTFEnactor <Problem>
        Enactor;

    Csr<VertexId, SizeT, Value> *csr    = info->csr_ptr;
    VertexId      src                   = info->info["source_vertex"     ].get_int64();
    int           max_grid_size         = info->info["max_grid_size"     ].get_int  ();
    int           num_gpus              = info->info["num_gpus"          ].get_int  ();
    double        max_queue_sizing      = info->info["max_queue_sizing"  ].get_real (); 
    double        max_queue_sizing1     = info->info["max_queue_sizing1" ].get_real (); 
    double        max_in_sizing         = info->info["max_in_sizing"     ].get_real (); 
    std::string   partition_method      = info->info["partition_method"  ].get_str  (); 
    double        partition_factor      = info->info["partition_factor"  ].get_real (); 
    int           partition_seed        = info->info["partition_seed"    ].get_int  (); 
    bool          quick_mode            = info->info["quick_mode"        ].get_bool ();
    bool          quiet_mode            = info->info["quiet_mode"        ].get_bool ();
    bool          stream_from_host      = info->info["stream_from_host"  ].get_bool ();
    bool          instrument            = info->info["instrument"        ].get_bool (); 
    bool          debug                 = info->info["debug_mode"        ].get_bool (); 
    bool          size_check            = info->info["size_check"        ].get_bool (); 
    Value         alpha                 = info->info["alpha"             ].get_real ();
    Value         delta                 = info->info["delta"             ].get_real ();
    Value         error                 = info->info["error"             ].get_real ();
    SizeT         max_iter              = info->info["max_iteration"     ].get_int  ();
    CpuTimer      cpu_timer;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side label array (for both reference and gpu-computed results)
    Value    *reference_rank    = (Value*)malloc(sizeof(Value) * csr->nodes);
    Value    *h_rank            = (Value*)malloc(sizeof(Value) * csr->nodes);
    VertexId *h_node_id         = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    VertexId *reference_node_id = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    Value    *reference_check   = (quick_mode) ? NULL : reference_rank;

    // Allocate problem on GPU
    Problem *problem = new Problem;
    util::GRError(problem -> Init(
        stream_from_host,
        csr,
        NULL,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "Problem WTF Initialization Failed", __FILE__, __LINE__);

    // Allocate WTF enactor map
    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor -> Init(
        context, problem, max_grid_size),
        "WTF Enactor Init failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // Perform WTF

    util::GRError(problem -> Reset(
        src, delta, alpha, error, enactor -> GetFrontierType(),
        max_queue_sizing, max_queue_sizing1),
        "WTF Problem Data Reset failed", __FILE__, __LINE__);
    util::GRError(enactor -> Reset(),
        "WTF Enactor Reset failed", __FILE__, __LINE__);

    cpu_timer.Start();
    util::GRError(enactor -> Enact(
         src, alpha, max_iter),
        "WTF Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();
    cpu_timer.Start();

    // Copy out results
    util::GRError(problem -> Extract(h_rank, h_node_id),
        "HITS Problem Data Extraction Failed", __FILE__, __LINE__);

    double total_pr = 0;
    for (SizeT i = 0; i < csr->nodes; ++i)
    {
        total_pr += h_rank[i];
    }

    //
    // Compute reference CPU HITS solution for source-distance
    //
    if (reference_check != NULL && total_pr > 0)
    {
        if (!quiet_mode) printf("compute ref value\n");
        ReferenceWTF(
            *csr,
            src,
            reference_node_id,
            reference_check,
            delta,
            alpha,
            max_iter);
        if (!quiet_mode) printf("\n");
    }

    // Verify the result
    if (reference_check != NULL && total_pr > 0)
    {
        if (!quiet_mode) printf("Validity: ");
        CompareResults(h_rank, reference_check, csr->nodes, true);
    }

    if (!quiet_mode)
    {
        printf("\nGPU result.");
        DisplaySolution(h_node_id, h_rank, csr->nodes);
    }

    info->ComputeCommonStats(enactor -> enactor_stats.GetPointer(), elapsed, (VertexId*)NULL);

    // Cleanup
    if (problem        ) delete problem;
    if (enactor        ) delete enactor;
    if (reference_check) free(reference_check);
    if (h_rank         ) free(h_rank);
    //cudaDeviceSynchronize();
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
}

/******************************************************************************
 * Main
 ******************************************************************************/
template <
    typename VertexId,  // use int as the vertex identifier
    typename SizeT   ,  // use int as the graph size type
    typename Value   >  // use int as the value type
int main_(CommandLineArgs *args)
{
    CpuTimer cpu_timer, cpu_timer2;
    cpu_timer.Start();

    //
    // Construct graph and perform search(es)
    //
    Csr <VertexId, SizeT, Value> csr(false); // default for stream_from_host
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    info->info["undirected"] = args -> CheckCmdLineFlag("undirected");
    cpu_timer2.Start();
    info->Init("WTF", *args, csr);
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    RunTests<VertexId, SizeT, Value>(info);

    cpu_timer.Stop();
    info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!(info->info["quiet_mode"].get_bool()))
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject
    return 0;
}

template <
    typename VertexId, // the vertex identifier type, usually int or long long
    typename SizeT   > // the size tyep, usually int or long long
int main_Value(CommandLineArgs *args)
{
// disabled to reduce compile time
//    if (args -> CheckCmdLineFlag("64bit-Value"))
//        return main_<VertexId, SizeT, double>(args);
//    else
        return main_<VertexId, SizeT, float >(args);
}

template <
    typename VertexId>
int main_SizeT(CommandLineArgs *args)
{
// disabled to reduce compile time
//    if (args -> CheckCmdLineFlag("64bit-SizeT"))
//        return main_Value<VertexId, long long>(args);
//    else
        return main_Value<VertexId, int      >(args);
}

int main_VertexId(CommandLineArgs *args)
{
    // disabled, because oprtr::filter::KernelPolicy::SmemStorage is too large for 64bit VertexId
    //if (args -> CheckCmdLineFlag("64bit-VertexId"))
    //    return main_SizeT<long long>(args);
    //else 
        return main_SizeT<int      >(args);
}

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    return main_VertexId(&args);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
