// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sysinfo.h>
#include <gunrock/util/json_spirit_writer_template.h>
#include <gunrock/util/gitsha1.h>
#include <boost/filesystem.hpp>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

/******************************************************************************
 * Macros
 ******************************************************************************/

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf(
        " test_bfs <graph type> <graph type args> [--device=<device_index>]\n"
        " [--undirected] [--src=<source_index>] [--idempotence=<0|1>] [--v]\n"
        " [--instrumented] [--iteration-num=<num>] [--traversal-mode=<0|1>]\n"
        " [--mark-pred] [--queue-sizing=<scaleFactor>] [--grid-size=<size>]\n"
        " [--in-sizing=<in/out queue scale factor>] [--disable-size-check]\n"
        " [--quick] [partition_method=<random|biasrandom|clustered|metis]\n"
        " [--quiet] [--json] [--jsonfile=<name>] [--jsondir=<dir>]"
        "\n"
        "Graph types and args:\n"
        "  market <file>\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed / undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<GPU index>    Set GPU device for running the test. [Default: 0].\n"
        "  --undirected            Treat the graph as undirected (symmetric).\n"
        "  --idempotence=<0|1>     Enable: 1, Disable: 0 [Default: Enable].\n"
        "  --instrumented          Keep kernels statics [Default: Disable].\n"
        "                          total_queued, search_depth and barrier duty\n"
        "                          (a relative indicator of load imbalance.)\n"
        "  --src=<vertex id>       Begins BFS from the source [Default: 0].\n"
        "                          If randomize: from a random source vertex.\n"
        "                          If largestdegree: from largest degree vertex.\n"
        "  --quick=<0 or 1>        Skip the CPU validation: 1, or not: 0 [Default: 1].\n"
        "  --mark-pred             Keep both label info and predecessor info.\n"
        "  --queue-sizing=<factor> Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). [Default: 1.0]\n"
        "  --v                     Print verbose per iteration debug info.\n"
        "  --iteration-num=<num>   Number of runs to perform the test [Default: 1].\n"
        "  --traversal-mode=<0|1>  Set traversal strategy, 0 for Load-Balanced, \n"
        "                          1 for Dynamic-Cooperative [Default: dynamic\n"
        "                          determine based on average degree].\n"
        " --quiet                  No output (unless --json is specified)\n"
        " --json                   Output JSON-format statistics to stdout\n"
        " --jsonfile=<name>        Output JSON-format statistics to file <name>\n"
        " --jsondir=<dir>          Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated\n"
    );
}

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] labels    Search depth from the source for each node.
 * @param[in] preds     Predecessor node id for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <
    typename VertexId,
    typename SizeT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void DisplaySolution(
    VertexId *labels,
    VertexId *preds,
    SizeT     num_nodes,
    bool quiet = false)
{
    if (quiet) { return; }
    // careful: if later code in this
    // function changes something, this
    // return is the wrong thing to do

    if (num_nodes > 40) { num_nodes = 40; }

    printf("\nFirst %d labels of the GPU result:\n", num_nodes);

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(labels[i]);
        if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE)
        {
            printf(",");
            PrintValue(preds[i]);
        }
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

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam MARK_PREDECESSORS
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] src Source node where BFS starts
 * @param[in] h_labels Host-side vector stores computed labels for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] search_depth Maximum search depth of the BFS algorithm
 * @param[in] total_queued Total element queued in BFS kernel running process
 * @param[in] avg_duty Average duty of the BFS kernels
 */
template <
    bool MARK_PREDECESSORS,
    typename VertexId,
    typename Value,
    typename SizeT >
void DisplayStats(
    Stats                &stats,
    json_spirit::mObject &info,
    VertexId             src,
    const VertexId       *h_labels,
    const Csr<VertexId, Value, SizeT> *graph,
    double               elapsed,
    VertexId             search_depth,
    long long            total_queued,
    double               avg_duty,
    bool                 quiet = false)
{
    // Compute nodes and edges visited
    SizeT edges_visited = 0;
    SizeT nodes_visited = 0;
    for (VertexId i = 0; i < graph->nodes; ++i)
    {
        if (h_labels[i] < util::MaxValue<VertexId>() && h_labels[i] != -1)
        {
            ++nodes_visited;
            edges_visited += graph->row_offsets[i + 1] - graph->row_offsets[i];
        }
    }

    double redundant_work = 0.0;
    if (total_queued > 0)
    {
        // measure duplicate edges put through queue
        redundant_work =
            ((double)total_queued - edges_visited) / edges_visited;
    }
    redundant_work *= 100;

    // Display test name
    if (!quiet)
    {
        printf("[%s] finished. ", stats.name);
    }

    // Display statistics
    if (nodes_visited < 5)
    {
        if (!quiet)
        {
            printf("Fewer than 5 vertices visited.\n");
        }
    }
    else
    {
        // Display the specific sample statistics
        double m_teps = (double) edges_visited / (elapsed * 1000.0);
        if (!quiet)
        {
            printf("\n elapsed: %.4f ms, rate: %.4f MiEdges/s", elapsed,
                   m_teps);
        }

        if (search_depth != 0)
        {
            if (!quiet)
            {
                printf(", search_depth: %lld", (long long) search_depth);
            }
        }

        if (avg_duty != 0)
        {
            if (!quiet)
            {
                printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
            }
        }

        if (!quiet)
        {
            printf("\n src: %lld, nodes_visited: %lld, edges_visited: %lld",
                   (long long) src, (long long) nodes_visited,
                   (long long) edges_visited);
        }

        if (total_queued > 0)
        {
            if (!quiet)
            {
                printf(", total queued: %lld", total_queued);
            }
        }

        if (redundant_work > 0)
        {
            if (!quiet)
            {
                printf(", redundant work: %.2f%%", redundant_work);
            }
        }

        info["name"] = stats.name;
        info["elapsed"] = elapsed;
        info["m_teps"] = m_teps;
        info["search_depth"] = search_depth;
        info["avg_duty"] = avg_duty;
        info["nodes_visited"] = nodes_visited;
        info["edges_visited"] = edges_visited;
        info["total_queued"] = int64_t(total_queued);
        info["redundant_work"] = redundant_work;

        if (!quiet)
        {
            printf("\n");
        }
    }
}

/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference BFS ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source_path Host-side vector to store CPU computed labels for each node
 * @param[in] predecessor Host-side vector to store CPU computed predecessor for each node
 * @param[in] src Source node where BFS starts
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void SimpleReferenceBfs(
    const Csr<VertexId, Value, SizeT> *graph,
    VertexId                          *source_path,
    VertexId                          *predecessor,
    VertexId                          src,
    bool                              quiet = false)
{
    // Initialize labels
    for (VertexId i = 0; i < graph->nodes; ++i)
    {
        source_path[i] = ENABLE_IDEMPOTENCE ? -1 : util::MaxValue<VertexId>() - 1;
        if (MARK_PREDECESSORS)
        {
            predecessor[i] = -1;
        }
    }
    source_path[src] = 0;
    VertexId search_depth = 0;

    // Initialize queue for managing previously-discovered nodes
    std::deque<VertexId> frontier;
    frontier.push_back(src);

    //
    // Perform BFS
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty())
    {
        // Dequeue node from frontier
        VertexId dequeued_node = frontier.front();
        frontier.pop_front();
        VertexId neighbor_dist = source_path[dequeued_node] + 1;

        // Locate adjacency list
        SizeT edges_begin = graph->row_offsets[dequeued_node];
        SizeT edges_end = graph->row_offsets[dequeued_node + 1];

        for (SizeT edge = edges_begin; edge < edges_end; ++edge)
        {
            //Lookup neighbor and enqueue if undiscovered
            VertexId neighbor = graph->column_indices[edge];
            if (source_path[neighbor] > neighbor_dist || source_path[neighbor] == -1)
            {
                source_path[neighbor] = neighbor_dist;
                if (MARK_PREDECESSORS)
                {
                    predecessor[neighbor] = dequeued_node;
                }
                if (search_depth < neighbor_dist)
                {
                    search_depth = neighbor_dist;
                }
                frontier.push_back(neighbor);
            }
        }
    }

    if (MARK_PREDECESSORS)
    {
        predecessor[src] = -1;
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    if (!quiet)
    {
        printf("CPU BFS finished in %lf msec. cpu_search_depth: %d\n",
               elapsed, search_depth);
    }
}

/**
 * @brief Run BFS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] src Source node where BFS starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] max_queue_sizing Scaling factor used in edge mapping
 * @param[in] iterations Number of iterations for running the test
 * @param[in] traversal_mode Advance mode: Load-balanced or Dynamic cooperative
 * @param[in] context CudaContext pointer for ModernGPU APIs
 *
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK,
    bool        MARK_PREDECESSORS,
    bool        ENABLE_IDEMPOTENCE >
void RunTests(app::Info<VertexId, Value, SizeT> *info)
{
    // info->json();
    typedef BFSProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS,
            ENABLE_IDEMPOTENCE,
            (MARK_PREDECESSORS && ENABLE_IDEMPOTENCE) >
            BfsProblem;  // does not use double buffer

    typedef BFSEnactor < BfsProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK >
            BfsEnactor;

    // parse configurations from mObject info
    Csr<VertexId, Value, SizeT> *graph = (Csr<VertexId, Value, SizeT>*)info->graph;
    VertexId src                 = info->info["vertex_id"].get_int64();
    int max_grid_size            = info->info["max_grid_size"].get_int();
    int num_gpus                 = info->info["num_gpus"].get_int();
    double max_queue_sizing      = info->info["queue_sizing"].get_real();
    double max_queue_sizing1     = info->info["queue_sizing1"].get_real();
    double max_in_sizing         = info->info["max_in_sizing"].get_real();
    std::string partition_method = info->info["partition_method"].get_str();
    float partition_factor       = info->info["partition_factor"].get_real();
    int partition_seed           = info->info["partition_seed"].get_int();
    bool quiet_mode              = info->info["quiet_mode"].get_bool();
    bool g_quick                 = info->info["quick_mode"].get_bool();
    bool g_stream_from_host      = info->info["stream_from_host"].get_bool();
    int traversal_mode           = info->info["traversal_mode"].get_int();
    int iterations               = info->info["iterations"].get_int();

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    ContextPtr   *context = (ContextPtr*)info -> context;  // TODO: remove after merge mgpu-cq
    cudaStream_t *streams = info -> streams;  // TODO: remove after merge mgpu-cq

    size_t       *org_size              = new size_t  [num_gpus];

    // Allocate host-side label array (for both reference and GPU results)
    VertexId     *reference_labels      = new VertexId[graph->nodes];
    VertexId     *reference_preds       = new VertexId[graph->nodes];
    VertexId     *h_labels              = new VertexId[graph->nodes];
    VertexId     *reference_check_label = (g_quick) ? NULL : reference_labels;
    VertexId     *reference_check_preds = NULL;
    VertexId     *h_preds               = NULL;

    if (MARK_PREDECESSORS)
    {
        h_preds = new VertexId[graph->nodes];
        if (!g_quick)
        {
            reference_check_preds = reference_preds;
        }
    }

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }
    // Allocate BFS enactor map
    BfsEnactor *enactor = new BfsEnactor(num_gpus, gpu_idx);

    // Allocate problem on GPU
    BfsProblem *problem = new BfsProblem;
    util::GRError(problem->Init(
                      g_stream_from_host,
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
                  "Problem BFS Initialization Failed", __FILE__, __LINE__);

    util::GRError(enactor->Init(context, problem, max_grid_size, traversal_mode),
                  "BFS Enactor Init failed", __FILE__, __LINE__);

    //
    // Compute reference CPU BFS solution for source-distance
    //

    if (reference_check_label != NULL)
    {
        if (!quiet_mode)
        {
            printf("Computing reference value ...\n");
        }
        SimpleReferenceBfs<VertexId, Value, SizeT,
                           MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>(
                               graph,
                               reference_check_label,
                               reference_check_preds,
                               src,
                               quiet_mode);
        if (!quiet_mode)
        {
            printf("\n");
        }
    }

    Stats     *stats       = new Stats("GPU BFS");
    long long total_queued = 0;
    VertexId  search_depth = 0;
    double    avg_duty     = 0.0;
    float     elapsed      = 0.0;

    //
    // Perform BFS
    //

    CpuTimer cpu_timer;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(src, enactor->GetFrontierType(),
                                     max_queue_sizing, max_queue_sizing1),
                      "BFS Problem Data Reset Failed", __FILE__, __LINE__);

        util::GRError(
            enactor->Reset(),
            "BFS Enactor Reset failed", __FILE__, __LINE__);

        util::GRError("Error before Enact", __FILE__, __LINE__);

        if (!quiet_mode)
        {   printf("__________________________\n"); fflush(stdout); }

        cpu_timer.Start();
        util::GRError(enactor->Enact(src, traversal_mode),
                      "BFS Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();

        if (!quiet_mode)
        {   printf("--------------------------\n"); fflush(stdout); }

        elapsed += cpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    enactor->GetStatistics(total_queued, search_depth, avg_duty);

    // Copy out results
    util::GRError(problem->Extract(h_labels, h_preds),
                  "BFS Problem Data Extraction Failed", __FILE__, __LINE__);

    // Verify the result
    if (reference_check_label != NULL)
    {
        if (!ENABLE_IDEMPOTENCE)
        {
            if (!quiet_mode)
            {
                printf("Label Validity: ");
            }
            int error_num = CompareResults(
                                h_labels, reference_check_label,
                                graph->nodes, true, quiet_mode);
            if (error_num > 0)
            {
                if (!quiet_mode)
                {
                    printf("%d errors occurred.\n", error_num);
                }
            }
        }
        else
        {
            if (!MARK_PREDECESSORS)
            {
                if (!quiet_mode)
                {
                    printf("Label Validity: ");
                }
                int error_num = CompareResults(
                                    h_labels, reference_check_label,
                                    graph->nodes, true, quiet_mode);
                if (error_num > 0)
                {
                    if (!quiet_mode)
                    {
                        printf("%d errors occurred.\n", error_num);
                    }
                }
            }
        }
    }

    // Display Solution
    if (!quiet_mode)
    {
        DisplaySolution <
        VertexId, SizeT, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE >
        (h_labels, h_preds, graph->nodes, quiet_mode);
    }

//    info->computeTraversalStats(stats, elapsed, h_labels, graph);

    DisplayStats<MARK_PREDECESSORS, VertexId, Value, SizeT>(
        *stats,
        info->info,
        src,
        h_labels,
        graph,
        elapsed,
        search_depth,
        total_queued,
        avg_duty,
        quiet_mode);

    if (!quiet_mode)
    {
        printf("\n\tMemory Usage(B)\t");
        for (int gpu = 0; gpu < num_gpus; gpu++)
            if (num_gpus > 1)
            {
                if (gpu != 0)
                {
                    printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1",
                           gpu, gpu, gpu, gpu);
                }
                else
                {
                    printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
                }
            }
            else
            {
                printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
            }
        if (num_gpus > 1)
        {
            printf(" #keys%d", num_gpus);
        }
        printf("\n");
        double max_queue_sizing_[2] = {0, 0 }, max_in_sizing_ = 0;
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
                    if (factor > max_queue_sizing_[j])
                    {
                        max_queue_sizing_[j] = factor;
                    }
                }
                if (num_gpus > 1 && i != 0 )
                {
                    for (int t = 0; t < 2; t++)
                    {
                        SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                        printf("\t %lld", (long long) x);
                        double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
                        if (factor > max_in_sizing_)
                        {
                            max_in_sizing_ = factor;
                        }
                    }
                }
            }
            if (num_gpus > 1)
            {
                printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
            }
            printf("\n");
        }
        printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
        if (num_gpus > 1)
        {
            printf("\t in_sizing =\t %lf", max_in_sizing_);
        }
        printf("\n");
    }

    // Clean up
    if (org_size        ) {delete[] org_size        ; org_size         = NULL;}
    if (stats           ) {delete   stats           ; stats            = NULL;}
    if (enactor         ) {delete   enactor         ; enactor          = NULL;}
    if (problem         ) {delete   problem         ; problem          = NULL;}
    if (reference_labels) {delete[] reference_labels; reference_labels = NULL;}
    if (reference_preds ) {delete[] reference_preds ; reference_preds  = NULL;}
    if (h_labels        ) {delete[] h_labels        ; h_labels         = NULL;}
    if (h_preds         ) {delete[] h_preds         ; h_preds          = NULL;}
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
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] info Pointer to mObject info.
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK,
    bool        MARK_PREDECESSORS >
void RunTests_enable_idempotence(app::Info<VertexId, Value, SizeT> *info)
{
    if (info->info["idempotent"].get_bool())
    {
        RunTests <VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
                 MARK_PREDECESSORS, true > (info);
    }
    else
    {
        RunTests <VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
                 MARK_PREDECESSORS, false> (info);
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
 * @param[in] info Pointer to mObject info.
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK >
void RunTests_mark_predecessors(app::Info<VertexId, Value, SizeT> *info)
{
    if (info->info["mark_preds"].get_bool())
    {
        RunTests_enable_idempotence<VertexId, Value, SizeT, INSTRUMENT,
                                    DEBUG, SIZE_CHECK,  true> (info);
    }
    else
    {
        RunTests_enable_idempotence<VertexId, Value, SizeT, INSTRUMENT,
                                    DEBUG, SIZE_CHECK, false> (info);
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
 *
 * @param[in] info Pointer to mObject info.
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG >
void RunTests_size_check(app::Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
    {
        RunTests_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT,
                                   DEBUG,  true> (info);
    }
    else
    {
        RunTests_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT,
                                   DEBUG, false> (info);
    }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] info Pointer to mObject info.
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT >
void RunTests_debug(app::Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
    {
        RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT,  true> (info);
    }
    else
    {
        RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT, false> (info);
    }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to mObject info.
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT >
void RunTests_instrumented(app::Info<VertexId, Value, SizeT> *info)
{
    if (info->info["instrument"].get_bool())
    {
        RunTests_debug < VertexId, Value, SizeT, true> (info);
    }
    else
    {
        RunTests_debug<VertexId, Value, SizeT,  false> (info);
    }
}

/******************************************************************************
* Main
******************************************************************************/

int main(int argc, char** argv)
{
    // command line check
    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    // define data types
    typedef int VertexId;  // Use int as the vertex identifier
    typedef int Value;     // Use int as the value type
    typedef int SizeT;     // Use int as the graph size type

    Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
    app::Info<VertexId, Value, SizeT> *info = new app::Info<VertexId, Value, SizeT>;

    info->Init(args, csr);  // initialize Info structure

    RunTests_instrumented<VertexId, Value, SizeT>(info);  // run test

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
