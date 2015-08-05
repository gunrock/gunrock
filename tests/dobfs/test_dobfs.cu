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
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// DOBFS includes
#include <gunrock/app/dobfs/dobfs_enactor.cuh>
#include <gunrock/app/dobfs/dobfs_problem.cuh>
#include <gunrock/app/dobfs/dobfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;
using namespace gunrock::app::dobfs;


/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf (
        " test_dobfs <graph type> <graph type args> [--device=<device_index>]\n"
        " [--src=<source_index>] [--instrumented] [--idempotence=<0|1>] [--v]\n"
        " [--undirected] [--iteration-num=<num>] [--quick=<0|1>] [--mark-pred]\n"
        " [--queue-sizing=<scale factor>]\n"
        "\n"
        "Graph types and args:\n"
        "  market <file>\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed / undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>   Set GPU device for running the test. [Default: 0].\n"
        "  --undirected              Treat the graph as undirected (symmetric).\n"
        "  --idempotence=<0 or 1>    Enable: 1, Disable: 0 [Default: Enable].\n"
        "  --instrumented            Keep kernels statics [Default: Disable].\n"
        "                            total_queued, search_depth and barrier duty\n"
        "                            (a relative indicator of load imbalance.)\n"
        "  --src=<source vertex id>  Begins BFS from the source [Default: 0].\n"
        "                            If randomize: from a random source vertex.\n"
        "                            If largestdegree: from largest degree vertex.\n"
        "  --quick=<0 or 1>          Skip the CPU validation: 1, or not: 0 [Default: 1].\n"
        "  --mark-pred               Keep both label info and predecessor info.\n"
        "  --queue-sizing=<factor>   Allocates a frontier queue sized at: \n"
        "                            (graph-edges * <scale factor>). [Default: 1.0]\n"
        "  --v                       Print verbose per iteration debug info.\n"
        "  --iteration-num=<number>  Number of runs to perform the test [Default: 1].\n"
    );
}

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] preds Predecessor node id for each node.
 * @param[in] nodes Number of nodes in the graph.
 * @param[in] MARK_PREDECESSORS Whether to show predecessor of each node.
 * @param[in] ENABLE_IDEMPOTENCE Whether to enable idempotence mode.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution(
    VertexId *source_path,
    VertexId *preds,
    SizeT nodes,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE)
{
    if (nodes > 40) nodes = 40;
    printf("\nFirst %d labels of the GPU result.\n", nodes);

    printf("[");
    for (VertexId i = 0; i < nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE)
        {
            printf(",");
            PrintValue(preds[i]);
        }
        printf(" ");
    }
    printf("]\n");
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
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source_path Host-side vector to store CPU computed labels for each node
 * @param[in] src Source node where BFS starts
 * @param[in] enable_idempotence Whether or not to enable idempotent
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void SimpleReferenceBfs(
    const Csr<VertexId, Value, SizeT>       &graph,
    VertexId                                *source_path,
    VertexId                                src,
    bool                                    enable_idempotence,
    bool                                    quiet = false)
{
    // Initialize distances
    if (enable_idempotence)
    {
        for (VertexId i = 0; i < graph.nodes; ++i)
        {
            source_path[i] = -1;
        }
    }
    else
    {
        for (VertexId i = 0; i < graph.nodes; ++i)
        {
            source_path[i] = util::MaxValue<VertexId>() - 1;
        }
    }
    source_path[src] = 0;
    VertexId search_depth = 0;

    // Initialize queue for managing previously-discovered nodes
    std::deque<VertexId> frontier;
    frontier.push_back(src);

    // Perform BFS
    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty())
    {

        // Dequeue node from frontier
        VertexId dequeued_node = frontier.front();
        frontier.pop_front();
        VertexId neighbor_dist = source_path[dequeued_node] + 1;

        // Locate adjacency list
        int edges_begin = graph.row_offsets[dequeued_node];
        int edges_end = graph.row_offsets[dequeued_node + 1];

        for (int edge = edges_begin; edge < edges_end; ++edge)
        {
            //Lookup neighbor and enqueue if undiscovered
            VertexId neighbor = graph.column_indices[edge];
            if (source_path[neighbor] == -1 ||
                    source_path[neighbor] == util::MaxValue<VertexId>() - 1)
            {
                source_path[neighbor] = neighbor_dist;
                if (search_depth < neighbor_dist)
                {
                    search_depth = neighbor_dist;
                }
                frontier.push_back(neighbor);
            }
        }
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    if (!quiet)
    {
        printf("CPU BFS finished in %lf msec. Search depth is: %d\n",
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
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
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
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void RunTests(Info<VertexId, Value, SizeT> *info)
{
    typedef DOBFSProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS,
            ENABLE_IDEMPOTENCE,
            (MARK_PREDECESSORS && ENABLE_IDEMPOTENCE) >
            Problem; // does not use double buffer

    Csr<VertexId, Value, SizeT> *csr = (Csr<VertexId, Value, SizeT>*)info->csr_ptr;
    Csr<VertexId, Value, SizeT> *csc = (Csr<VertexId, Value, SizeT>*)info->csc_ptr;
    VertexId src            = info->info["source_vertex"].get_int64();
    int max_grid_size       = info->info["max_grid_size"].get_int();
    int num_gpus            = info->info["num_gpus"].get_int();
    double max_queue_sizing = info->info["max_queue_sizing"].get_real();
    bool quiet_mode         = info->info["quiet_mode"].get_bool();
    bool quick_mode         = info->info["quick_mode"].get_bool();
    bool undirected         = info->info["undirected"].get_bool();
    bool stream_from_host   = info->info["stream_from_host"].get_bool();
    int iterations          = info->info["num_iteration"].get_int();
    double alpha            = info->info["alpha"].get_real();
    double beta             = info->info["beta"].get_real();

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr *context = (ContextPtr*)info->context;

    // allocate host-side label array (for both reference and GPU results)
    VertexId *reference_labels = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    VertexId *h_labels         = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    VertexId *reference_check  = (quick_mode) ? NULL : reference_labels;
    VertexId *h_preds          = NULL;
    if (MARK_PREDECESSORS)
    {
        h_preds = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    }

    // Allocate BFS enactor map
    DOBFSEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> enactor(gpu_idx);

    // Allocate problem on GPU
    Problem *problem = new Problem;

    util::GRError(problem->Init(
                      stream_from_host,
                      undirected,
                      *csr,
                      *csc,
                      num_gpus,
                      alpha,
                      beta),
                  "Problem DOBFS Initialization Failed", __FILE__, __LINE__);

    // compute reference CPU BFS solution
    if (reference_check != NULL)
    {
        if (!quiet_mode) { printf(" computing reference value ...\n"); }
        SimpleReferenceBfs(
            *csr,
            reference_check,
            src,
            ENABLE_IDEMPOTENCE,
            quiet_mode);
        if (!quiet_mode) { printf("\n"); }
    }

    double elapsed = 0.0f;

    // Perform BFS
    GpuTimer gpu_timer;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(
                          src, enactor.GetFrontierType(), max_queue_sizing),
                      "DOBFS Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(enactor.template Enact<Problem>(
                          *context, problem, src, max_grid_size),
                      "DOBFS Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();
        elapsed += gpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    // Copy out results
    util::GRError(problem->Extract(h_labels, h_preds),
                  "DOBFS Problem Data Extraction Failed", __FILE__, __LINE__);

    // Verify the result
    if (reference_check != NULL)
    {
        if (!MARK_PREDECESSORS)
        {
            if (!quiet_mode) { printf("Validity: "); }
            CompareResults(
                h_labels, reference_check, csr->nodes, true, quiet_mode);
        }
    }

    if (!quiet_mode)
    {
        DisplaySolution(h_labels, h_preds, csr->nodes,
                        MARK_PREDECESSORS, ENABLE_IDEMPOTENCE);
    }

    info->ComputeTraversalStats(  // compute running statistics
        enactor.enactor_stats.GetPointer(), elapsed, h_labels);

    if (!quiet_mode)
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject

    // Clean up
    if (problem) delete problem;
    if (reference_labels) free(reference_labels);
    if (h_labels) free(h_labels);
    if (h_preds) free(h_preds);

    cudaDeviceSynchronize();
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
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK,
    bool        MARK_PREDECESSORS >
void RunTests_enable_idempotence(Info<VertexId, Value, SizeT> *info)
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
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK >
void RunTests_mark_predecessors(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["mark_predecessors"].get_bool())
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
    {
        RunTests_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT,
                                   DEBUG,  true>(info);
    }
    else
    {
        RunTests_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT,
                                   DEBUG, false>(info);
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
    {
        RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT,  true>(info);
    }
    else
    {
        RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT, false>(info);
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
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT >
void RunTests_instrumented(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["instrument"].get_bool())
    {
        RunTests_debug<VertexId, Value, SizeT, true>(info);
    }
    else
    {
        RunTests_debug<VertexId, Value, SizeT, false>(info);
    }
}

/******************************************************************************
* Main
******************************************************************************/

int main( int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    typedef int VertexId;  // Use int as the vertex identifier
    typedef int Value;     // Use int as the value type
    typedef int SizeT;     // Use int as the graph size type

    Csr<VertexId, Value, SizeT> csr(false);  // CSR graph we process on
    Csr<VertexId, Value, SizeT> csc(false);  // CSC graph we process on
    Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info->info["undirected"] = args.CheckCmdLineFlag("undirected");

    info->Init("DOBFS", args, csr, csc);  // initialize Info structure
    RunTests_instrumented<VertexId, Value, SizeT>(info);  // run test

    return 0;
}
