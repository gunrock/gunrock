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
 * @brief Simple test driver program for computing Pagerank.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/rgg.cuh>

// BFS includes
#include <gunrock/app/pr/pr_enactor.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/app/pr/pr_functor.cuh>

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
using namespace gunrock::app::pr;


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
        "[--instrumented]          Keep kernels statics [Default: Disable].\n"
        "                          total_queued, search_depth and barrier duty.\n"
        "                          (a relative indicator of load imbalance.)\n"
        "[--quick]                 Skip the CPU reference validation process.\n"
        "[--disable-size-check]    Disable frontier queue size check.\n"
        "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
        "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--in-sizing=<in/out_queue_scale_factor>]\n"
        "                          Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--max-iter=<num>]        Max iteration for rank score distribution\n"
        "                          before one round of PageRank run end.\n"
        "[--partition_method=<random|biasrandom|clustered|metis>]\n"
        "                          Choose partitioner (Default use random).\n"
        "[--delta=<delta>]         Delta for PageRank (Default 0.85f).\n"
        "[--error=<error>]         Error threshold for PageRank (Default 0.01f).\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays the PageRank result
 *
 * @param[in] node Node vertex Id
 * @param[in] rank Rank value for the node
 * @param[in] nodes Number of nodes in the graph.
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(VertexId *node, Value *rank, SizeT nodes)
{
    int top = (nodes < 10) ? nodes : 10;  // at most top 10 ranked nodes
    printf("\nTop %d Ranked Vertices and PageRanks:\n", top);
    for (int i = 0; i < top; ++i)
    {
        printf("Vertex ID: %d, PageRank: %.8le\n", node[i], (double)rank[i]);
    }
}

/**
 * @brief Compares the equivalence of two arrays. If incorrect, print the location
 * of the first incorrect value appears, the incorrect value, and the reference
 * value.
 *
 * @tparam T datatype of the values being compared with.
 * @tparam SizeT datatype of the array length.
 *
 * @param[in] computed Vector of values to be compared.
 * @param[in] reference Vector of reference values
 * @param[in] len Vector length
 * @param[in] verbose Whether to print values around the incorrect one.
 * @param[in] quiet     Don't print out anything to stdout
 *
 * \return Zero if two vectors are exactly the same, non-zero if there is any difference.
 */
template <typename SizeT, typename Value>
int CompareResults_(
    Value* computed,
    Value* reference,
    SizeT len,
    bool verbose = true,
    bool quiet = false,
    Value threshold = 0.05f)
{
    int flag = 0;
    for (SizeT i = 0; i < len; i++)
    {

        // Use relative error rate here.
        bool is_right = true;
        if (fabs(computed[i]) < 0.01f && fabs(reference[i] - 1) < 0.01f) continue;
        if (fabs(computed[i] - 0.0) < 0.01f)
        {
            if (fabs(computed[i] - reference[i]) > threshold)
                is_right = false;
        }
        else
        {
            if (fabs((computed[i] - reference[i]) / reference[i]) > threshold)
                is_right = false;
        }
        if (!is_right && flag == 0)
        {
            if (!quiet)
            {
                printf("\nINCORRECT: [%lu]: ", (unsigned long) i);
                PrintValue<Value>(computed[i]);
                printf(" != ");
                PrintValue<Value>(reference[i]);

                if (verbose)
                {
                    printf("\nresult[...");
                    for (SizeT j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++)
                    {
                        PrintValue<Value>(computed[j]);
                        printf(", ");
                    }
                    printf("...]");
                    printf("\nreference[...");
                    for (SizeT j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++)
                    {
                        PrintValue<Value>(reference[j]);
                        printf(", ");
                    }
                    printf("...]");
                }
            }
            flag += 1;
        }
        if (!is_right && flag > 0) flag += 1;
    }
    if (!quiet)
    {
        printf("\n");
        if (!flag)
        {
            printf("CORRECT");
        }
    }
    return flag;
}

/******************************************************************************
 * PageRank Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference Page Rank implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_id Source node for personalized PageRank (if any)
 * @param[in] rank Host-side vector to store CPU computed labels for each node
 * @param[in] delta Delta for computing PR
 * @param[in] error Error threshold
 * @param[in] max_iteration Maximum iteration to go
 * @param[in] directed Whether the graph is directed
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void SimpleReferencePageRank(
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId                          *node_id,
    Value                             *rank,
    Value                             delta,
    Value                             error,
    SizeT                             max_iteration,
    bool                              directed,
    bool                              quiet = false)
{
    using namespace boost;

    // preparation
    typedef adjacency_list< vecS, vecS, bidirectionalS, no_property,
            property<edge_index_t, int> > Graph;

    Graph g;

    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
        {
            Graph::edge_descriptor e =
                add_edge(i, graph.column_indices[j], g).first;
            put(edge_index, g, e, i);
        }
    }

    // compute PageRank
    CpuTimer cpu_timer;
    cpu_timer.Start();

    std::vector<Value> ranks(num_vertices(g));
    page_rank(g, make_iterator_property_map(
                  ranks.begin(),
                  get(boost::vertex_index, g)),
              boost::graph::n_iterations(max_iteration));

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    for (std::size_t i = 0; i < num_vertices(g); ++i)
    {
        rank[i] = ranks[i];
    }

    // Sort the top ranked vertices
    RankPair<SizeT, Value> *pr_list =
        (RankPair<SizeT, Value>*)malloc(
            sizeof(RankPair<SizeT, Value>) * num_vertices(g));
    for (int i = 0; i < num_vertices(g); ++i)
    {
        pr_list[i].vertex_id = i;
        pr_list[i].page_rank = rank[i];
    }
    std::stable_sort(pr_list, pr_list + num_vertices(g),
                     PRCompare<RankPair<SizeT, Value> >);

    for (int i = 0; i < num_vertices(g); ++i)
    {
        node_id[i] = pr_list[i].vertex_id;
        rank[i] = pr_list[i].page_rank;
    }

    free(pr_list);
    if (!quiet) { printf("CPU PageRank finished in %lf msec.\n", elapsed); }
}

/*template <
    typename VertexId,
    typename Value>
class Sort_Pair {
public:
    VertexId v;
    Value val;
};

template <
    typename VertexId,
    typename Value>
inline bool operator< (const Sort_Pair<VertexId, Value>& lhs, const Sort_Pair<VertexId, Value>& rhs)
{
    if (lhs.val < rhs.val) return true;
    if (rhs.val < lhs.val) return false;
    return false;
}*/

/**
 * @brief A simple CPU-based reference Page Rank implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_id Source node for personalized PageRank (if any)
 * @param[in] rank Host-side vector to store CPU computed labels for each node
 * @param[in] delta Delta for computing PR
 * @param[in] error Error threshold
 * @param[in] max_iteration Maximum iteration to go
 * @param[in] directed Whether the graph is directed
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void SimpleReferencePageRank_Normalized(
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId                          *node_id,
    Value                             *rank,
    Value                             delta,
    Value                             error,
    SizeT                             max_iteration,
    bool                              directed,
    bool                              quiet = false,
    bool                              scaled = false)
{
    //typedef Sort_Pair<VertexId, Value> SPair;
    SizeT nodes = graph.nodes;
    //SizeT edges = graph.edges;
    Value *rank_current = (Value*) malloc (sizeof(Value) * nodes);
    Value *rank_next    = (Value*) malloc (sizeof(Value) * nodes);
    //SPair *sort_pairs   = (SPair*) malloc (sizeof(SPair) * nodes);
    bool  to_continue   = true;
    SizeT iteration     = 0;
    Value reset_value   = scaled ? 1.0 - delta : ((1.0 - delta) / (Value)nodes);
    CpuTimer cpu_timer;

    cpu_timer.Start();
    //#pragma omp parallel
    {
        #pragma omp parallel for
        for (VertexId v=0; v<nodes; v++)
        {
            rank_current[v] = scaled ? 1.0 : (1.0 / (Value)nodes);
            rank_next   [v] = 0;
        }

        while (to_continue)
        {
            to_continue = false;

            #pragma omp parallel for
            for (VertexId src=0; src<nodes; src++)
            {
                SizeT start_e = graph.row_offsets[src];
                SizeT end_e   = graph.row_offsets[src+1];
                if (start_e == end_e) continue; // 0 out degree vertex
                Value dist_rank = rank_current[src] / (Value)(end_e - start_e);
                if (!isfinite(dist_rank)) continue;
                for (SizeT e = start_e; e < end_e; e++)
                {
                    VertexId dest = graph.column_indices[e];
                    #pragma omp atomic
                        rank_next[dest] += dist_rank;
                }
            }

            iteration ++;

            #pragma omp parallel for
            for (VertexId v=0; v<nodes; v++)
            {
                Value rank_new = delta * rank_next[v];
                if (!isfinite(rank_new)) rank_new = 0;
                rank_new = rank_new + reset_value;
                if (iteration <= max_iteration &&
                    fabs(rank_new - rank_current[v]) > error * rank_current[v])
                {
                    to_continue = true;
                }
                rank_current[v] = rank_new;
                rank_next   [v] = 0;
            }

            //#pragma omp single
            //{
            //    iteration ++;
            //}
        }    
    }
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    // Sort the top ranked vertices
    RankPair<SizeT, Value> *pr_list =
        (RankPair<SizeT, Value>*)malloc(
            sizeof(RankPair<SizeT, Value>) * nodes);

    #pragma omp parallel for
    for (VertexId i = 0; i < nodes; ++i)
    {
        pr_list[i].vertex_id = i;
        pr_list[i].page_rank = rank_current[i];
    }

    std::stable_sort(pr_list, pr_list + nodes, 
                     PRCompare<RankPair<SizeT, Value> >);

    #pragma omp parallel for
    for (VertexId i = 0; i < nodes; ++i)
    {
        node_id[i] = pr_list[i].vertex_id;
        rank[i] = scaled ? (pr_list[i].page_rank / (Value)nodes) : pr_list[i].page_rank;
    }

    free(pr_list     ); pr_list      = NULL;
    free(rank_current); rank_current = NULL;
    free(rank_next   ); rank_next    = NULL;
    if (!quiet) 
    {
        printf("CPU iteration : %lld\n", (long long)iteration); 
        printf("CPU PageRank finished in %lf msec.\n", elapsed); 
    }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK,
    bool NORMALIZED>
void RunTests(Info<VertexId, Value, SizeT> *info)
{
    typedef PRProblem <VertexId,
            SizeT,
            Value,
            NORMALIZED> PrProblem;

    typedef PREnactor <PrProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK > PrEnactor;

    // parse configurations from mObject info
    Csr<VertexId, Value, SizeT> *graph = info->csr_ptr;
    VertexId src                 = info->info["source_vertex"].get_int64();
    bool undirected              = info->info["undirected"].get_bool();
    bool quiet_mode              = info->info["quiet_mode"].get_bool();
    bool quick_mode              = info->info["quick_mode"].get_bool();
    bool stream_from_host        = info->info["stream_from_host"].get_bool();
    int max_grid_size            = info->info["max_grid_size"].get_int();
    int num_gpus                 = info->info["num_gpus"].get_int();
    int max_iteration            = info->info["max_iteration"].get_int();
    double max_queue_sizing      = info->info["max_queue_sizing"].get_real();
    double max_queue_sizing1     = info->info["max_queue_sizing1"].get_real();
    double max_in_sizing         = info->info["max_in_sizing"].get_real();
    std::string partition_method = info->info["partition_method"].get_str();
    double partition_factor      = info->info["partition_factor"].get_real();
    int partition_seed           = info->info["partition_seed"].get_int();
    int iterations               = 1; //force to 1 info->info["num_iteration"].get_int();
    int traversal_mode           = info->info["traversal_mode"].get_int();
    std::string ref_filename     = info->info["ref_filename"].get_str();
    Value delta                  = info->info["delta"].get_real();
    Value error                  = info->info["error"].get_real();
    bool  scaled                 = info->info["scaled"].get_bool();
    CpuTimer cpu_timer;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side array (for both reference and GPU-computed results)
    Value        *ref_rank           = new Value   [graph->nodes];
    Value        *h_rank             = new Value   [graph->nodes];
    VertexId     *h_node_id          = new VertexId[graph->nodes];
    VertexId     *ref_node_id        = new VertexId[graph->nodes];
    //Value        *ref_check          = (quick_mode) ? NULL : ref_rank;

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    PrEnactor* enactor = new PrEnactor(num_gpus, gpu_idx);  // enactor map
    PrProblem *problem = new PrProblem;  // allocate problem on GPU
    problem -> scaled  = scaled;

    util::GRError(problem->Init(
                      stream_from_host,
                      graph,
                      NULL,
                      num_gpus,
                      gpu_idx,
                      partition_method,
                      streams,
                      max_queue_sizing,
                      max_in_sizing,
                      partition_factor,
                      partition_seed),
                  "PR Problem Init failed", __FILE__, __LINE__);
    util::GRError(enactor->Init(
                      context, problem, traversal_mode, max_grid_size),
                  "PR Enactor Init failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    double elapsed = 0.0f;

    // perform PageRank

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(
                          src, delta, error, max_iteration,
                          enactor->GetFrontierType(), max_queue_sizing,
                          max_queue_sizing1, traversal_mode == 1 ? true : false),
                      "PR Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(enactor->Reset(),
                      "PR Enactor Reset Reset failed", __FILE__, __LINE__);

        if (!quiet_mode)
        {
            printf("__________________________\n"); fflush(stdout);
        }
        cpu_timer.Start();
        util::GRError(enactor->Enact(traversal_mode),
                      "PR Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        if (!quiet_mode)
        {
            printf("--------------------------\n"); fflush(stdout);
        }
        elapsed += cpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    cpu_timer.Start();
    // copy out results
    util::GRError(problem->Extract(h_rank, h_node_id),
                  "PR Problem Data Extraction Failed", __FILE__, __LINE__);

    if (!quiet_mode)
    {
        double total_pr = 0;
        for (SizeT i = 0; i < graph->nodes; ++i)
        {
            total_pr += h_rank[i];
        }
        printf("Total rank : %.10lf\n", total_pr);
    }

    // compute reference CPU solution
    if (!quick_mode)
    {
        if (!quiet_mode) { printf("Computing reference value ...\n"); }
        if (NORMALIZED)
            SimpleReferencePageRank_Normalized <VertexId, Value, SizeT>(
                *graph,
                ref_node_id,
                ref_rank,
                delta,
                error,
                max_iteration,
                !undirected,
                quiet_mode,
                scaled);
        else SimpleReferencePageRank <VertexId, Value, SizeT>(
                *graph,
                ref_node_id,
                ref_rank,
                delta,
                error,
                max_iteration,
                !undirected,
                quiet_mode);
        if (!quiet_mode) { printf("\n"); }

        // Verify the result
        if (!quiet_mode) { printf("Validity Rank: \n"); }
        Value *unorder_rank = new Value[graph->nodes];
        int   *v_count      = new int  [graph->nodes];
        SizeT  error_count  = 0;
        for (VertexId i=0; i<graph->nodes; i++)
            v_count[i] = 0;

        for (VertexId i=0; i<graph->nodes; i++)
        {
            VertexId v = h_node_id[i];
            if (v < 0 || v >= graph->nodes)
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : node_id[%d] (%d) is out of bound\n", i, v);
                error_count ++;
                continue;
            }
            if (v_count[v] > 0)
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : node_id[%d] (%d) appears more than once\n", i, v);
                error_count ++;
                continue;
            }
            v_count[v] ++;
            unorder_rank[v] = h_rank[i];
        }
        for (VertexId v=0; v<graph->nodes; v++)
        if (v_count[v] == 0)
        {
            if (error_count == 0 && !quiet_mode)
                printf("INCORRECT : vertex %d does not appear in result\n", v);
            error_count ++;
        }
        double ref_total_rank = 0;
        double max_diff       = 0;
        VertexId max_diff_pos = graph->nodes;
        double max_rdiff      = 0;
        VertexId max_rdiff_pos= graph->nodes;
        for (VertexId i=0; i<graph->nodes; i++)
        {
            VertexId v = ref_node_id[i];
            ref_total_rank += ref_rank[i];
            Value diff = fabs(ref_rank[i] - unorder_rank[v]);
            if ((ref_rank[i] > 1e-12 && diff > error * ref_rank[i]) ||
                (ref_rank[i] <= 1e-12 && diff > error))
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : rank[%d] (%.8le) != %.8le\n",
                        v, (double)unorder_rank[v], (double)ref_rank[i]);
                error_count ++;
            }
            if (diff > max_diff)
            {
                max_diff = diff;
                max_diff_pos = i;
            }
            if (ref_rank[i] > 1e-12)
            {
                Value rdiff = diff / ref_rank[i];
                if (rdiff > max_rdiff)
                {
                    max_rdiff = rdiff;
                    max_rdiff_pos = i;
                }
            }
        }
        if (error_count == 0 && !quiet_mode)
            printf("CORRECT\n");
        else if (!quiet_mode)
            printf("number of errors : %lld\n", (long long) error_count);
        printf("Reference total rank : %.10lf\n", ref_total_rank);
        printf("Maximum difference : "); 
        if (max_diff_pos < graph->nodes)
            printf("rank[%d] %.8le vs. %.8le, ",
                ref_node_id[max_diff_pos], 
                (double)unorder_rank[ref_node_id[max_diff_pos]], 
                (double)ref_rank[max_diff_pos]);
        printf("%.8le\n", (double)max_diff);
        printf("Maximum relative difference :");
        if (max_rdiff_pos < graph->nodes)
            printf("rank[%d] %.8le vs. %.8le, ",
                ref_node_id[max_rdiff_pos],
                (double)unorder_rank[ref_node_id[max_rdiff_pos]],
                (double)ref_rank[max_rdiff_pos]);
        printf("%.8lf %%\n", (double)max_rdiff * 100);

        if (!quiet_mode) { printf("Validity Order: \n"); }
        error_count = 0;
        for (SizeT i=0; i<graph->nodes-1; i++)
        if (h_rank[i] < h_rank[i+1])
        {
            if (error_count == 0 && !quiet_mode)
                printf("INCORRECT : rank[%lld] (%.8le), place %lld < rank[%lld] (%.8le), place %lld\n",
                    (long long)h_node_id[i  ], (double)h_rank[i  ], (long long)i, 
                    (long long)h_node_id[i+1], (double)h_rank[i+1], (long long)i+1);
            error_count ++;
        }
        if (error_count == 0 && !quiet_mode)
            printf("CORRECT\n");
        else if (!quiet_mode)
            printf("number of errors : %lld\n", (long long) error_count);
        delete[] unorder_rank; unorder_rank = NULL;

        /*SizeT errors_count = CompareResults_(
                           h_rank, ref_check,
                           graph->nodes, true, quiet_mode, error);
        if (errors_count > 0)
        {
            if (!quiet_mode)
            {
                printf("number of errors : %lld\n", (long long) errors_count);
            }
        }*/
    }

    if (!quiet_mode)
    {
        //printf("\nFirst 40 labels of the GPU result.");
        // Display Solution
        DisplaySolution(h_node_id, h_rank, graph->nodes);
    }

    info->ComputeCommonStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), elapsed, NULL, true);

    if (!quiet_mode)
    {
        printf("\n\tMemory Usage(B)\t");
        for (int gpu = 0; gpu < num_gpus; gpu++)
            if (num_gpus > 1) {if (gpu != 0) printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1", gpu, gpu, gpu, gpu); else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);}
            else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
        if (num_gpus > 1) printf(" #keys%d", num_gpus);
        printf("\n");
        double max_queue_sizing_[2] = {0, 0}, max_in_sizing_ = 0;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            size_t gpu_free, dummy;
            cudaSetDevice(gpu_idx[gpu]);
            cudaMemGetInfo(&gpu_free, &dummy);
            printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
            for (int i = 0; i < num_gpus; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    SizeT x = problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
                    printf("\t %lld", (long long) x);
                    double factor = 1.0 * x / (num_gpus > 1 ? problem->graph_slices[gpu]->in_counter[i] : problem->graph_slices[gpu]->nodes);
                    if (factor > max_queue_sizing_[j]) max_queue_sizing_[j] = factor;
                }
                if (num_gpus > 1 && i != 0 )
                    for (int t = 0; t < 2; t++)
                    {
                        SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                        printf("\t %lld", (long long) x);
                        double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
                        if (factor > max_in_sizing_) max_in_sizing_ = factor;
                    }
            }
            if (num_gpus > 1) printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
            printf("\n");
        }
        printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
        if (num_gpus > 1) printf("\t in_sizing =\t %lf", max_in_sizing_);
        printf("\n");
    }

    // Clean up
    if (org_size   ) { delete   org_size   ; org_size    = NULL; }
    if (problem    ) { delete   problem    ; problem     = NULL; }
    if (enactor    ) { delete   enactor    ; enactor     = NULL; }
    if (ref_rank   ) { delete[] ref_rank   ; ref_rank    = NULL; }
    if (ref_node_id) { delete[] ref_node_id; ref_node_id = NULL; }
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    
    if (h_rank     ) 
    {
        if (info->info["output_filename"].get_str() !="")
        {
            cpu_timer.Start();
            std::ofstream fout;
            size_t buf_size = 1024 * 1024 * 16;
            char *fout_buf = new char[buf_size];
            fout.rdbuf() -> pubsetbuf(fout_buf, buf_size);
            fout.open(info->info["output_filename"].get_str().c_str());

            for (VertexId i=0; i<graph->nodes; i++)
            {
                fout<< h_node_id[i]+1 << "," << h_rank[i] << std::endl;
            }
            fout.close();
            delete[] fout_buf; fout_buf = NULL;
            cpu_timer.Stop();
            info->info["write_time"] = cpu_timer.ElapsedMillis();
        }
        delete[] h_rank     ; h_rank      = NULL; 
    }
    if (h_node_id  ) { delete[] h_node_id  ; h_node_id   = NULL; }
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    
    if (h_rank     ) 
    {
        if (info->info["output_filename"].get_str() !="")
        {
            cpu_timer.Start();
            std::ofstream fout;
            size_t buf_size = 1024 * 1024 * 16;
            char *fout_buf = new char[buf_size];
            fout.rdbuf() -> pubsetbuf(fout_buf, buf_size);
            fout.open(info->info["output_filename"].get_str().c_str());

            for (VertexId v=0; v<graph->nodes; v++)
            {
                fout<< v+1 << "," << h_rank[v] << std::endl;
            }
            fout.close();
            delete[] fout_buf; fout_buf = NULL;
            cpu_timer.Stop();
            info->info["write_time"] = cpu_timer.ElapsedMillis();
        }
        delete[] h_rank     ; h_rank      = NULL; 
    }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK >
void RunTests_normalized(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["normalized"].get_bool())
        RunTests<VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,  true>(info);
    else
        RunTests<VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK, false>(info);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG >
void RunTests_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
        RunTests_normalized<VertexId, Value, SizeT, INSTRUMENT, DEBUG,  true>(info);
    else
        RunTests_normalized<VertexId, Value, SizeT, INSTRUMENT, DEBUG, false>(info);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT >
void RunTests_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
        RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT,  true>(info);
    else
        RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT, false>(info);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT >
void RunTests_instrumented(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["instrument"].get_bool())
        RunTests_debug<VertexId, Value, SizeT,  true>(info);
    else
        RunTests_debug<VertexId, Value, SizeT, false>(info);
}

/******************************************************************************
 * Main
 ******************************************************************************/

template<
    typename VertexId,
    typename Value,
    typename SizeT>
int main_(CommandLineArgs *args)
{
    CpuTimer cpu_timer, cpu_timer2;

    cpu_timer.Start();
    //typedef int VertexId;  // use int as the vertex identifier
    //typedef float Value;   // use float as the value type
    //typedef int SizeT;     // use int as the graph size type

    Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
    Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    if (args -> CheckCmdLineFlag("normalized"))
        info->info["undirected"] = args -> CheckCmdLineFlag("undirected");
    else info->info["undirected"] = true;   // require undirected input graph when unnormalized

    cpu_timer2.Start();
    info->Init("PageRank", *args, csr);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    RunTests_instrumented<VertexId, Value, SizeT>(info);  // run test

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
    typename VertexId,
    typename Value>   // the value type, usually int or long long
int main_SizeT(CommandLineArgs *args)
{
    if (args -> CheckCmdLineFlag("64bit-SizeT"))
        return main_<VertexId, Value, long long>(args);
    else 
        return main_<VertexId, Value, int      >(args);
}

template <
    typename VertexId> // the vertex identifier type, usually int or long long
int main_Value(CommandLineArgs *args)
{
    if (args -> CheckCmdLineFlag("64bit-Value"))
        return main_SizeT<VertexId, double>(args);
    else 
        return main_SizeT<VertexId, float >(args);
}

int main_VertexId(CommandLineArgs *args)
{
    //if (args -> CheckCmdLineFlag("64bit-VertexId"))
    //    return main_Value<long long>(args);
    //else 
        return main_Value<int      >(args);
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
