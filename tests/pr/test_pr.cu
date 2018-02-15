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

// PR includes
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

#include <gunrock/util/shared_utils.cuh>

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
        "[--partition-method=<random|biasrandom|clustered|metis>]\n"
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
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(VertexId *node, Value *rank, SizeT nodes)
{
    SizeT top = (nodes < 10) ? nodes : 10;  // at most top 10 ranked nodes
    printf("\nTop %lld Ranked Vertices and PageRanks:\n", (long long)top);
    for (SizeT i = 0; i < top; ++i)
    {
        printf("Vertex ID: %lld, PageRank: %.8le\n", (long long)node[i], (double)rank[i]);
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
 * @param[in] reference Vector of reference values.
 * @param[in] len Vector length.
 * @param[in] verbose Whether to print values around the incorrect one.
 * @param[in] quiet     Don't print out anything to stdout.
 * @param[in] threshold Results error checking threshold.
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
    typename SizeT,
    typename Value >
void ReferencePageRank(
    const Csr<VertexId, SizeT, Value> &graph,
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
 * @param[in] scaled Normalized flag
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value >
void ReferencePageRank_Normalized(
    const Csr<VertexId, SizeT, Value> &graph,
    VertexId                          *node_id,
    Value                             *rank,
    Value                             delta,
    Value                             error,
    SizeT                             max_iteration,
    bool                              directed,
    bool                              quiet = false,
    bool                              scaled = false)
{
    SizeT nodes = graph.nodes;
    Value *rank_current = (Value*) malloc (sizeof(Value) * nodes);
    Value *rank_next    = (Value*) malloc (sizeof(Value) * nodes);
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
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool NORMALIZED>
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info)
{
    typedef PRProblem <VertexId,
            SizeT,
            Value,
            NORMALIZED> Problem;

    typedef PREnactor <Problem>
            //INSTRUMENT,
            //DEBUG,
            //SIZE_CHECK >
            Enactor;

    // parse configurations from mObject info
    Csr<VertexId, SizeT, Value> *graph = info->csr_ptr;
    VertexId    src                 = info->info["source_vertex"    ].get_int64();
    bool        undirected          = info->info["undirected"       ].get_bool ();
    bool        quiet_mode          = info->info["quiet_mode"       ].get_bool ();
    bool        quick_mode          = info->info["quick_mode"       ].get_bool ();
    bool        stream_from_host    = info->info["stream_from_host" ].get_bool ();
    int         max_grid_size       = info->info["max_grid_size"    ].get_int  ();
    int         num_gpus            = info->info["num_gpus"         ].get_int  ();
    int         max_iteration       = info->info["max_iteration"    ].get_int  ();
    double      max_queue_sizing    = 0.0; //info->info["max_queue_sizing" ].get_real ();
    double      max_queue_sizing1   = 0.0; //info->info["max_queue_sizing1"].get_real ();
    double      max_in_sizing       = 1.0; //info->info["max_in_sizing"    ].get_real ();
    std::string partition_method    = info->info["partition_method" ].get_str  ();
    double      partition_factor    = info->info["partition_factor" ].get_real ();
    int         partition_seed      = info->info["partition_seed"   ].get_int  ();
    bool        instrument          = info->info["instrument"       ].get_bool ();
    bool        debug               = info->info["debug_mode"       ].get_bool ();
    bool        size_check          = info->info["size_check"       ].get_bool ();
    int         iterations          = info->info["num_iteration"    ].get_int  ();
    std::string traversal_mode      = info->info["traversal_mode"   ].get_str  ();

    std::string ref_filename        = info->info["ref_filename"     ].get_str  ();
    Value       delta               = info->info["delta"            ].get_real ();
    Value       error               = info->info["error"            ].get_real ();
    bool        scaled              = info->info["scaled"           ].get_bool ();
    bool        compensate          = info->info["compensate"       ].get_bool ();
    int      communicate_latency    = info->info["communicate_latency"].get_int ();
    float    communicate_multipy    = info->info["communicate_multipy"].get_real();
    int      expand_latency         = info->info["expand_latency"    ].get_int ();
    int      subqueue_latency       = info->info["subqueue_latency"  ].get_int ();
    int      fullqueue_latency      = info->info["fullqueue_latency" ].get_int ();
    int      makeout_latency        = info->info["makeout_latency"   ].get_int ();
    if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;

    CpuTimer    cpu_timer;
    cudaError_t retval              = cudaSuccess;
    
    if (traversal_mode == "LB_CULL")
    {
	printf("Traversal Mode LB_CULL not available for PageRank\n");
	exit(0);
    }    

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        if (retval = cudaSetDevice(gpu_idx[gpu])) return retval;
        if (retval = cudaMemGetInfo(&(org_size[gpu]), &dummy)) return retval;
    }

    if (compensate)
    {
        util::Array1D<SizeT, VertexId> zero_out_vertices;

        zero_out_vertices.Allocate(graph -> nodes, util::HOST);
        SizeT counter = 0;
        for (VertexId v = 0; v< graph->nodes; v++)
        if (graph -> row_offsets[v+1] == graph -> row_offsets[v])
        {
            zero_out_vertices[counter] = v;
            counter ++;
        }
        if (counter != 0)
        {
            if (!quiet_mode) printf("Adding 1 vertex and %lld edges to compensate 0 degree vertices\n",
                (long long)counter + (long long)graph -> nodes);
            util::Array1D<SizeT, VertexId> new_column_indices;
            util::Array1D<SizeT, SizeT   > new_row_offsets;
            new_column_indices.Allocate(graph -> edges + counter + graph -> nodes, util::HOST);
            new_row_offsets   .Allocate(graph -> nodes + 2);
            SizeT edge_counter = 0;
            for (VertexId v = 0; v < graph->nodes; v++)
            {
                new_row_offsets[v] = edge_counter;
                if (graph -> row_offsets[v+1] == graph -> row_offsets[v])
                {
                    new_column_indices[edge_counter] = graph -> nodes;
                    edge_counter ++;
                } else {
                    SizeT num_neighbors = graph -> row_offsets[v+1] - graph -> row_offsets[v];
                    for (SizeT e = 0; e < num_neighbors; e++)
                        new_column_indices[edge_counter + e] = graph -> column_indices[graph -> row_offsets[v] + e];
                    edge_counter += num_neighbors;
                }
            }
            for (VertexId v = 0; v< graph -> nodes; v++)
                new_column_indices[edge_counter + v] = v;
            new_row_offsets[graph -> nodes] = edge_counter;
            edge_counter += graph -> nodes;
            new_row_offsets[graph -> nodes + 1] = edge_counter;
            free(graph -> column_indices);
            graph -> column_indices = (VertexId*) malloc((long long)edge_counter * sizeof(VertexId));
            memcpy(graph -> column_indices, new_column_indices.GetPointer(util::HOST),
                sizeof(VertexId) * (long long)edge_counter);
            new_column_indices.Release();
            free(graph -> row_offsets);
            graph -> row_offsets = (SizeT*) malloc (((long long)graph -> nodes + 2) * sizeof(SizeT));
            memcpy(graph -> row_offsets, new_row_offsets.GetPointer(util::HOST),
                sizeof(SizeT) * ((long long)graph -> nodes + 2));
            graph -> edges = edge_counter;
            graph -> nodes +=1;
        }
    }

    // Allocate host-side array (for both reference and GPU-computed results)
    Value        *ref_rank           = new Value   [graph->nodes];
    Value        *h_rank             = new Value   [graph->nodes];
    VertexId     *h_node_id          = new VertexId[graph->nodes];
    VertexId     *ref_node_id        = new VertexId[graph->nodes];
    //Value        *ref_check          = (quick_mode) ? NULL : ref_rank;

    Problem *problem = new Problem(scaled);  // allocate problem on GPU
    if (retval = util::GRError(problem->Init(
        stream_from_host,
        graph,
        NULL,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        context,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "PR Problem Init failed", __FILE__, __LINE__))
        return retval;

    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);  // enactor map
    if (retval = util::GRError(enactor->Init(
        context, problem, traversal_mode, max_grid_size),
        "PR Enactor Init failed", __FILE__, __LINE__))
        return retval;

    enactor -> communicate_latency = communicate_latency;
    enactor -> communicate_multipy = communicate_multipy;
    enactor -> expand_latency      = expand_latency;
    enactor -> subqueue_latency    = subqueue_latency;
    enactor -> fullqueue_latency   = fullqueue_latency;
    enactor -> makeout_latency     = makeout_latency;

    if (retval = util::SetDevice(gpu_idx[0])) return retval;
    if (retval = util::latency::Test(
        streams[0], problem -> data_slices[0] -> latency_data,
        communicate_latency,
        communicate_multipy,
        expand_latency,
        subqueue_latency,
        fullqueue_latency,
        makeout_latency)) return retval;

    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // perform PageRank
    double total_elapsed = 0.0;
    double single_elapsed = 0.0;
    double max_elapsed    = 0.0;
    double min_elapsed    = 1e10;
    json_spirit::mArray process_times;
    if (!quiet_mode) printf("Using traversal mode %s\n", traversal_mode.c_str());

    for (int iter = 0; iter < iterations; ++iter)
    {
        if (retval = util::GRError(problem->Reset(
            src, delta, error, max_iteration,
            enactor->GetFrontierType(), max_queue_sizing,
            max_queue_sizing1, traversal_mode == "TWC" ? true : false),
            "PR Problem Data Reset Failed", __FILE__, __LINE__))
            return retval;
        if (retval = util::GRError(enactor->Reset(traversal_mode),
            "PR Enactor Reset Reset failed", __FILE__, __LINE__))
            return retval;

        if (!quiet_mode)
        {
            printf("__________________________\n"); fflush(stdout);
        }
        cpu_timer.Start();
        if (retval = util::GRError(enactor->Enact(traversal_mode),
            "PR Problem Enact Failed", __FILE__, __LINE__))
            return retval;
        cpu_timer.Stop();

        single_elapsed = cpu_timer.ElapsedMillis();
        total_elapsed += single_elapsed;
        process_times.push_back(single_elapsed);
        if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
        if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;
        if (!quiet_mode)
        {
            printf("--------------------------\n"
                "iteration %d elapsed: %lf ms\n",
                iter, single_elapsed);
            fflush(stdout);
        }
    }
    total_elapsed /= iterations;
    info -> info["process_times"] = process_times;
    info -> info["min_process_time"] = min_elapsed;
    info -> info["max_process_time"] = max_elapsed;

    cpu_timer.Start();
    // copy out results
    if (retval = util::GRError(enactor->Extract(),
        "PR Enactor extract failed", __FILE__, __LINE__))
        return retval;
    if (retval = util::GRError(problem->Extract(h_rank, h_node_id),
        "PR Problem Data Extraction Failed", __FILE__, __LINE__))
        return retval;

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
            ReferencePageRank_Normalized <VertexId, SizeT, Value>(
                *graph,
                ref_node_id,
                ref_rank,
                delta,
                error,
                max_iteration,
                !undirected,
                quiet_mode,
                scaled);
        else ReferencePageRank <VertexId, SizeT, Value>(
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
        SizeT *v_count      = new SizeT[graph->nodes];
        SizeT  error_count  = 0;
        for (VertexId i=0; i<graph->nodes; i++)
            v_count[i] = 0;

        for (VertexId i=0; i<graph->nodes; i++)
        {
            VertexId v = h_node_id[i];
            if (v < 0 || v >= graph->nodes)
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : node_id[%lld] (%lld) is out of bound\n",
                        (long long)i, (long long)v);
                error_count ++;
                continue;
            }
            if (v_count[v] > 0)
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : node_id[%lld] (%lld) appears more than once\n",
                        (long long)i, (long long)v);
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
                printf("INCORRECT : vertex %lld does not appear in result\n", (long long)v);
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
            if (v < 0 || v >= graph->nodes)
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : ref_node_id[%lld] = %lld, out of bound\n",
                        (long long)i, (long long)v);
                error_count ++;
                continue;
            }

            ref_total_rank += ref_rank[i];
            Value diff = fabs(ref_rank[i] - unorder_rank[v]);
            if ((ref_rank[i] > 1e-12 && diff > error * ref_rank[i]) ||
                (ref_rank[i] <= 1e-12 && diff > error))
            {
                if (error_count == 0 && !quiet_mode)
                    printf("INCORRECT : rank[%lld] (%.8le) != %.8le\n",
                        (long long)v, (double)unorder_rank[v], (double)ref_rank[i]);
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
        fflush(stdout);
        printf("Maximum difference : ");
        if (max_diff_pos < graph->nodes)
            printf("rank[%lld] %.8le vs. %.8le, ",
                (long long)ref_node_id[max_diff_pos],
                (double)unorder_rank[ref_node_id[max_diff_pos]],
                (double)ref_rank[max_diff_pos]);
        printf("%.8le\n", (double)max_diff);
        printf("Maximum relative difference :");
        if (max_rdiff_pos < graph->nodes)
            printf("rank[%lld] %.8le vs. %.8le, ",
                (long long)ref_node_id[max_rdiff_pos],
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
        enactor->enactor_stats.GetPointer(), total_elapsed, (VertexId*)NULL, true);

    /*if (!quiet_mode)
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
    }*/

    if (!quiet_mode)
    {
        Display_Memory_Usage(num_gpus, gpu_idx, org_size, problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
        Display_Performance_Profiling(enactor);
#endif
    }

    // Clean up
    if (org_size   ) { delete[] org_size   ; org_size    = NULL; }
    if (enactor         )
    {
        if (retval = util::GRError(enactor -> Release(),
            "BFS Enactor Release failed", __FILE__, __LINE__))
            return retval;
        delete   enactor         ; enactor          = NULL;
    }
    if (problem         )
    {
        if (retval = util::GRError(problem -> Release(),
            "BFS Problem Release failed", __FILE__, __LINE__))
            return retval;
        delete   problem         ; problem          = NULL;
    }
    if (ref_rank   ) { delete[] ref_rank   ; ref_rank    = NULL; }
    if (ref_node_id) { delete[] ref_node_id; ref_node_id = NULL; }
    if (gpu_idx    ) { delete[] gpu_idx    ; gpu_idx     = NULL; }
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

    return retval;
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
cudaError_t RunTests_normalized(Info<VertexId, SizeT, Value> *info)
{
    if (info->info["normalized"].get_bool())
        return RunTests<VertexId, SizeT, Value, true>(info);
    else
        return RunTests<VertexId, SizeT, Value, false>(info);
}

/******************************************************************************
 * Main
 ******************************************************************************/

template<
    typename VertexId,
    typename SizeT,
    typename Value>
int main_(CommandLineArgs *args)
{
    cudaError_t retval = cudaSuccess;
    CpuTimer cpu_timer, cpu_timer2;

    cpu_timer.Start();

    Csr <VertexId, SizeT, Value> csr(false);  // graph we process on
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    if (args -> CheckCmdLineFlag("normalized"))
         info->info["undirected"] = args -> CheckCmdLineFlag("undirected");
    else info->info["undirected"] = true;   // require undirected input graph when unnormalized

    cpu_timer2.Start();
    info->Init("PageRank", *args, csr);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    retval = RunTests_normalized<VertexId, SizeT, Value>(info);  // run test

    cpu_timer.Stop();
    info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!(info->info["quiet_mode"].get_bool()))
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject
    if (info) {delete info; info=NULL;}
    return retval;
}

template <
    typename VertexId, // the vertex identifier type, usually int or long long
    typename SizeT>
int main_Value(CommandLineArgs *args)
{
// can be disabled to reduce compile time
//    if (args -> CheckCmdLineFlag("64bit-Value"))
//        return main_<VertexId, SizeT, double>(args);
//    else
        return main_<VertexId, SizeT, float >(args);
}


template <
    typename VertexId>
int main_SizeT(CommandLineArgs *args)
{
// can be disabled to reduce compile time
    if (args -> CheckCmdLineFlag("64bit-SizeT") || sizeof(VertexId) > 4)
        return main_Value<VertexId, long long>(args);
    else
        return main_Value<VertexId, int      >(args);
}

int main_VertexId(CommandLineArgs *args)
{
    // can be disabled to reduce compile time
    if (args -> CheckCmdLineFlag("64bit-VertexId"))
        return main_SizeT<long long>(args);
    else
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
