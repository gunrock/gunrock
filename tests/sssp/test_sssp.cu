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

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes for CPU dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sssp;


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
    printf(
        " test_sssp <graph type> <graph type args> [--device=<device_index>]\n"
        " [--undirected] [--instrumented] [--src=<source index>] [--quick=<0|1>]\n"
        " [--mark-pred] [--queue-sizing=<scale factor>] [--traversal-mode=<0|1>]\n"
        " [--v] [--iteration-num=<num>]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed / undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>   Set GPU device for running the test. [Default: 0].\n"
        "  --undirected              Treat the graph as undirected (symmetric).\n"
        "  --instrumented            Keep kernels statics [Default: Disable].\n"
        "                            total_queued, search_depth and barrier duty\n"
        "                            (a relative indicator of load imbalance.)\n"
        "  --src=<source vertex id>  Begins SSSP from the source [Default: 0].\n"
        "                            If randomize: from a random source vertex.\n"
        "                            If largestdegree: from largest degree vertex.\n"
        "  --quick=<0 or 1>          Skip the CPU validation: 1, or not: 0 [Default: 1].\n"
        "  --mark-pred               Keep both label info and predecessor info.\n"
        "  --queue-sizing=<factor>   Allocates a frontier queue sized at:\n"
        "                            (graph-edges * <scale factor>) [Default: 1.0].\n"
        "  --v                       Print verbose per iteration debug info.\n"
        "  --iteration-num=<number>  Number of runs to perform the test [Default: 1].\n"
        "  --traversal-mode=<0 or 1> Set traversal strategy, 0 for Load-Balanced,\n"
        "                            1 for Dynamic-Cooperative [Default: dynamic\n"
        "                            determine based on average degree].\n"
        );
}

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] nodes Number of nodes in the graph.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution (VertexId *source_path, SizeT num_nodes)
{
    if (num_nodes > 40) num_nodes = 40;

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        printf(" ");
    }
    printf("]\n");
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
 * @tparam MARK_PREDECESSORS
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] src Source node where SSSP starts
 * @param[in] h_labels Host-side vector stores computed labels for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] search_depth Maximum search depth of the SSSP algorithm
 * @param[in] total_queued Total element queued in SSSP kernel running process
 * @param[in] avg_duty Average duty of the SSSP kernels
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void DisplayStats(
    Stats               &stats,
    VertexId            src,
    Value               *h_labels,
    const Csr<VertexId, Value, SizeT> &graph,
    double              elapsed,
    VertexId            search_depth,
    long long           total_queued,
    double              avg_duty)
{
    // Compute nodes and edges visited
    SizeT edges_visited = 0;
    SizeT nodes_visited = 0;
    for (VertexId i = 0; i < graph.nodes; ++i)
    {
        if (h_labels[i] < UINT_MAX)
        {
            ++nodes_visited;
            edges_visited += graph.row_offsets[i+1] - graph.row_offsets[i];
        }
    }

    double redundant_work = 0.0;
    if (total_queued > 0)
    {
        redundant_work =
            ((double) total_queued - edges_visited) / edges_visited;
    }
    redundant_work *= 100;

    // Display test name
    printf("[%s] finished.", stats.name);

    // Display statistics
    if (nodes_visited < 5)
    {
        printf("Fewer than 5 vertices visited.\n");
    }
    else
    {
        // Display the specific sample statistics
        double m_teps = (double) edges_visited / (elapsed * 1000.0);
        printf("\n elapsed: %.4f ms, rate: %.4f MiEdges/s", elapsed, m_teps);
        if (search_depth != 0)
            printf(", search_depth: %lld", (long long) search_depth);
        printf("\n src: %lld, nodes_visited: %lld, edges_visited: %lld",
               (long long) src, (long long) nodes_visited, (long long) edges_visited);
        if (avg_duty != 0 && g_verbose)
        {
            printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
        }
        if (total_queued > 0 && g_verbose)
        {
            printf(", total queued: %lld", total_queued);
        }
        if (redundant_work > 0 && g_verbose)
        {
            printf(", redundant work: %.2f%%", redundant_work);
        }
        printf("\n");
    }
}

/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference SSSP ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_values Host-side vector to store CPU computed labels for each node
 * @param[in] node_preds Host-side vector to store CPU computed predecessors for each node
 * @param[in] src Source node where SSSP starts
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT,
    bool     MARK_PREDECESSORS>
void SimpleReferenceSssp(
    const Csr<VertexId, Value, SizeT> &graph,
    Value                             *node_values,
    VertexId                          *node_preds,
    VertexId                          src)
{
    using namespace boost;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS, no_property,
                           property <edge_weight_t, unsigned int> > Graph;

    typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typedef graph_traits<Graph>::edge_descriptor edge_descriptor;

    typedef std::pair<VertexId, VertexId> Edge;

    Edge   *edges = ( Edge*)malloc(sizeof( Edge)*graph.edges);
    Value *weight = (Value*)malloc(sizeof(Value)*graph.edges);

    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
        {
            edges[j] = Edge(i, graph.column_indices[j]);
            weight[j] = graph.edge_values[j];
        }
    }

    Graph g(edges, edges + graph.edges, weight, graph.nodes);

    std::vector<Value> d(graph.nodes);
    std::vector<vertex_descriptor> p(graph.nodes);
    vertex_descriptor s = vertex(src, g);

    property_map<Graph, vertex_index_t>::type indexmap = get(vertex_index, g);

    //
    // Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    if (MARK_PREDECESSORS) {
        dijkstra_shortest_paths(g, s,
            predecessor_map(boost::make_iterator_property_map(
                    p.begin(), get(boost::vertex_index, g))).distance_map(
                        boost::make_iterator_property_map(
                            d.begin(), get(boost::vertex_index, g))));
    } else {
        dijkstra_shortest_paths(g, s,
            distance_map(boost::make_iterator_property_map(
                    d.begin(), get(boost::vertex_index, g))));
    }
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printf("CPU SSSP finished in %lf msec.\n", elapsed);

    Coo<Value, Value>* sort_dist = NULL;
    Coo<VertexId, VertexId>* sort_pred = NULL;
    sort_dist = (Coo<Value, Value>*)malloc(
        sizeof(Coo<Value, Value>) * graph.nodes);
    if (MARK_PREDECESSORS) {
        sort_pred = (Coo<VertexId, VertexId>*)malloc(
            sizeof(Coo<VertexId, VertexId>) * graph.nodes);
    }
    graph_traits < Graph >::vertex_iterator vi, vend;
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    {
        sort_dist[(*vi)].row = (*vi);
        sort_dist[(*vi)].col = d[(*vi)];
    }
    std::stable_sort(
        sort_dist, sort_dist + graph.nodes,
        RowFirstTupleCompare<Coo<Value, Value> >);

    if (MARK_PREDECESSORS)
    {
        for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
        {
            sort_pred[(*vi)].row = (*vi);
            sort_pred[(*vi)].col = p[(*vi)];
        }
        std::stable_sort(
            sort_pred, sort_pred + graph.nodes,
            RowFirstTupleCompare< Coo<VertexId, VertexId> >);
    }

    for (int i = 0; i < graph.nodes; ++i)
    {
        node_values[i] = sort_dist[i].col;
    }
    if (MARK_PREDECESSORS) {
        for (int i = 0; i < graph.nodes; ++i)
        {
            node_preds[i] = sort_pred[i].col;
        }
    }
    if (sort_dist) free(sort_dist);
    if (sort_pred) free(sort_pred);
}

/**
 * @brief Run SSSP tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] src Source node where SSSP starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] queue_sizing Scaling factor used in edge mapping
 * @param[in] num_gpus Number of GPUs
 * @param[in] delta_factor Parameter to specify delta in delta-stepping SSSP
 * @param[in] iterations Number of iteration for running the test
 & @param[in] traversal_mode Load-balanced or Dynamic cooperative
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool MARK_PREDECESSORS>
void RunTests(
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId src,
    int max_grid_size,
    float queue_sizing,
    int num_gpus,
    int delta_factor,
    int iterations,
    int traversal_mode,
    CudaContext& context)
{
    typedef SSSPProblem<
        VertexId,
        SizeT,
        Value,
        MARK_PREDECESSORS> Problem;

    // Allocate host-side arrays (for both reference and gpu-computed results)
    Value    *reference_labels = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value    *h_labels         = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value    *reference_check_label = (g_quick) ? NULL : reference_labels;
    VertexId *reference_preds       = NULL;
    VertexId *h_preds               = NULL;
    VertexId *reference_check_pred  = NULL;

    if (MARK_PREDECESSORS)
    {
        reference_preds = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
        h_preds         = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
        reference_check_pred  = (g_quick) ? NULL : reference_preds;
    }

    // Allocate SSSP enactor map
    SSSPEnactor<INSTRUMENT> sssp_enactor(g_verbose);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      g_stream_from_host,
                      graph,
                      num_gpus,
                      delta_factor),
                  "Problem SSSP Initialization Failed", __FILE__, __LINE__);

    //
    // Compute reference CPU SSSP solution for source-distance
    //
    if (reference_check_label != NULL)
    {
        printf("Computing reference value ...\n");
        SimpleReferenceSssp<VertexId, Value, SizeT, MARK_PREDECESSORS>(
            graph,
            reference_check_label,
            reference_check_pred,
            src);
        printf("\n");
    }

    Stats *stats = new Stats("GPU SSSP");

    long long           total_queued = 0;
    VertexId            search_depth = 0;
    double              avg_duty = 0.0;

    // Perform SSSP
    GpuTimer gpu_timer;

    float elapsed = 0.0f;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(
            csr_problem->Reset(
                src, sssp_enactor.GetFrontierType(), queue_sizing),
            "SSSP Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(
            sssp_enactor.template Enact<Problem>(
                context, csr_problem, src, queue_sizing,
                traversal_mode, max_grid_size),
            "SSSP Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();

        elapsed += gpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    sssp_enactor.GetStatistics(total_queued, search_depth, avg_duty);

    // Copy out results
    util::GRError(
        csr_problem->Extract(h_labels, h_preds),
        "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    // Display Solution
    printf("\nFirst 40 labels of the GPU result.\n");
    DisplaySolution(h_labels, graph.nodes);

    // Verify the result
    if (reference_check_label != NULL)
    {
        printf("Label Validity: ");
        CompareResults(h_labels, reference_check_label, graph.nodes, true);

        printf("\nFirst 40 labels of the reference CPU result.\n");
        DisplaySolution(reference_check_label, graph.nodes);
    }

    if (MARK_PREDECESSORS)
    {
        printf("\nFirst 40 preds of the GPU result.\n");
        DisplaySolution(h_preds, graph.nodes);

        if (reference_check_label != NULL)
        {
            printf("\nFirst 40 preds of the reference CPU result"
                   " (could be different because the paths are not unique).\n");
            DisplaySolution(reference_check_pred, graph.nodes);
        }
    }

    DisplayStats(
        *stats,
        src,
        h_labels,
        graph,
        elapsed,
        search_depth,
        total_queued,
        avg_duty);


    // Clean up
    if (stats)            delete stats;
    if (csr_problem)      delete csr_problem;
    if (reference_labels) free(reference_labels);
    if (h_labels)         free(h_labels);
    if (reference_preds)  free(reference_preds);
    if (h_preds)          free(h_preds);

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
    VertexId    src              = -1;  // Use whatever the specified graph-type's default is
    std::string src_str;
    bool        instrumented     = 0;   // Whether or not to collect instrumentation from kernels
    int         max_grid_size    = 0;   // Maximum grid size (0: leave it up to the enactor)
    int         num_gpus         = 1;   // Number of GPUs for multi-gpu enactor to use
    float       max_queue_sizing = 1.0; // Max queue sizing factor
    bool        mark_pred        = 0;   // Mark predecessor
    int         iterations       = 1;   // Number of runs for testing
    int         delta_factor     = 16;  // Delta factor for priority queue
    int         traversal_mode   = -1;  // traversal mode: 0 for LB, 1 for TWC
    g_quick                      = 0;   // Whether or not to skip ref validation

    // source vertex to start
    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty())
    {
        src = 0;
    }
    else if (src_str.compare("randomize") == 0)
    {
        src = graphio::RandomNode(graph.nodes);
    }
    else if (src_str.compare("largestdegree") == 0)
    {
        int max_degree;
        src = graph.GetNodeWithHighestDegree(max_degree);
        printf("Using highest degree (%d) vertex: %d\n", max_degree, src);
    }
    else
    {
        args.GetCmdLineArgument("src", src);
    }

    // traversal mode
    args.GetCmdLineArgument("traversal-mode", traversal_mode);
    if (traversal_mode == -1)
    {
        traversal_mode = graph.GetAverageDegree() > 8 ? 0 : 1;
    }

    instrumented = args.CheckCmdLineFlag("instrumented");
    mark_pred = args.CheckCmdLineFlag("mark-pred");
    g_verbose = args.CheckCmdLineFlag("v");
    g_quick   = args.CheckCmdLineFlag("quick");

    args.GetCmdLineArgument("iteration-num", iterations);
    args.GetCmdLineArgument("queue-sizing", max_queue_sizing);
    args.GetCmdLineArgument("delta-factor", delta_factor);

    // printf("Display neighbor list of src:\n");
    // graph.DisplayNeighborList(src);

    if (mark_pred) {
        if (instrumented) {
            RunTests<VertexId, Value, SizeT, true, true>(
                graph,
                src,
                max_grid_size,
                max_queue_sizing,
                num_gpus,
                delta_factor,
                iterations,
                traversal_mode,
                context);
        } else {
            RunTests<VertexId, Value, SizeT, false, true>(
                graph,
                src,
                max_grid_size,
                max_queue_sizing,
                num_gpus,
                delta_factor,
                iterations,
                traversal_mode,
                context);
        }
    } else {
        if (instrumented) {
            RunTests<VertexId, Value, SizeT, true, false>(
                graph,
                src,
                max_grid_size,
                max_queue_sizing,
                num_gpus,
                delta_factor,
                iterations,
                traversal_mode,
                context);
        } else {
            RunTests<VertexId, Value, SizeT, false, false>(
                graph,
                src,
                max_grid_size,
                max_queue_sizing,
                num_gpus,
                delta_factor,
                iterations,
                traversal_mode,
                context);
        }
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

    int dev = 0;
    args.GetCmdLineArgument("device", dev);
    ContextPtr context = mgpu::CreateCudaDevice(dev);

    // Parse graph-contruction params
    g_undirected = args.CheckCmdLineFlag("undirected");
    std::string graph_type = argv[1];
    int flags = args.ParsedArgc();
    int graph_args = argc - flags - 1;

    if (graph_args < 1) {
        Usage();
        return 1;
    }

    //
    // Construct graph and perform search(es)
    //

    if (graph_type == "market") {

        // Matrix-market coordinate-formatted graph file

        typedef int VertexId;                   // Use as the node identifier
        typedef unsigned int Value;             // Use as the value type
        typedef int SizeT;                      // Use as the graph size type

        Csr<VertexId, Value, SizeT> csr(false); // default for stream_from_host

        if (graph_args < 1) { Usage(); return 1; }
        char *market_filename = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<true>(
                market_filename,
                csr,
                g_undirected,
                false) != 0) // no inverse graph
        {
            return 1;
        }

        csr.PrintHistogram();
        csr.DisplayGraph(true); //print graph with edge_value
        //csr.GetAverageEdgeValue();
        //csr.GetAverageDegree();
        //int max_degree;
        //csr.GetNodeWithHighestDegree(max_degree);
        //printf("Max degree: %d\n", max_degree);

        // Run tests
        RunTests(csr, args, *context);

    } else {

        // Unknown graph type
        fprintf(stderr, "Unspecified graph type\n");
        return 1;

    }

    return 0;
}
