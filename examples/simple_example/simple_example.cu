// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * simple_example.cu
 *
 * @brief Simple test driver program for connected component.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// CC includes
#include <gunrock/app/cc/cc_enactor.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// BC includes
#include <gunrock/app/bc/bc_enactor.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

// Boost includes for CPU reference algorithms
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::cc;
using namespace gunrock::app::bfs;
using namespace gunrock::app::bc;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

template <typename VertexId>
struct CcList
{
    VertexId        root;
    unsigned int    histogram;

    CcList(VertexId root, unsigned int histogram) :
        root(root), histogram(histogram) {}
};

template<typename CcList>
bool CCCompare(
    CcList elem1,
    CcList elem2)
{
    return elem1.histogram > elem2.histogram;
}

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf(
        "\n test_cc <graph type> <graph type args> [--device=<device_index>]\n"
        " [--instrumented] [--quick] [--v] [--queue-sizing=<scale factor>]\n"
        " [--in-sizing=<in/out queue scale factor>] [--disable-size-check]\n"
        " [--partition_method=<random|biasrandom|clustered|metis]\n"
        " [--grid-sizing=<grid size>] [--partition_seed=<seed>]\n"
        " [--quiet] [--json] [--jsonfile=<name>] [--jsondir=<dir>]"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
        "  --instrumented If set then kernels keep track of queue-search_depth\n"
        "  and barrier duty (a relative indicator of load imbalance.)\n"
        "  --quick If set will skip the CPU validation code. Default: 0.\n"
        " --quiet                  No output (unless --json is specified).\n"
        " --json                   Output JSON-format statistics to stdout.\n"
        " --jsonfile=<name>        Output JSON-format statistics to file <name>\n"
        " --jsondir=<dir>          Output JSON-format statistics to <dir>/name,\n"
        
    );
}

/**
 * @brief Displays the CC result (i.e., number of components)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] comp_ids Host-side vector to store computed component id for each node
 * @param[in] nodes Number of nodes in the graph
 * @param[in] num_components Number of connected components in the graph
 * @param[in] roots Host-side vector stores the root for each node in the graph
 * @param[in] histogram Histogram of connected component ids
 */
template<typename VertexId, typename SizeT>
VertexId DisplayCCSolution(
    VertexId     *comp_ids,
    SizeT        nodes,
    unsigned int num_components,
    VertexId     *roots,
    unsigned int *histogram)
{
    typedef CcList<VertexId> CcListType;
    //printf("Number of components: %d\n", num_components);
    VertexId largest_cc_id;

    //sort the components by size
    CcListType *cclist =
        (CcListType*)malloc(sizeof(CcListType) * num_components);
    for (int i = 0; i < num_components; ++i)
    {
        cclist[i].root = roots[i];
        cclist[i].histogram = histogram[i];
    }
    std::stable_sort(
            cclist, cclist + num_components, CCCompare<CcListType>);

    // Print out at most top 10 largest components
    int top = (num_components < 10) ? num_components : 10;
    printf("Top %d largest components:\n", top);
    for (int i = 0; i < top; ++i)
    {
        printf("CC ID: %d, CC Root: %d, CC Size: %d\n",
                i, cclist[i].root, cclist[i].histogram);
    }
    largest_cc_id = cclist[0].root;

    free(cclist);
    return largest_cc_id;
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
 * @param[in] quiet     Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename SizeT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void DisplayBFSSolution(
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
 * @brief Displays the BC result (sigma value and BC value)
 *
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] sigmas
 * @param[in] bc_values
 * @param[in] nodes
 */
template<typename Value, typename SizeT>
void DisplayBCSolution(Value *sigmas, Value *bc_values, SizeT nodes)
{
    if (nodes < 40)
    {
        printf("[");
        for (SizeT i = 0; i < nodes; ++i)
        {
            PrintValue(i);
            printf(":");
            PrintValue(sigmas[i]);
            printf(",");
            PrintValue(bc_values[i]);
            printf(" ");
        }
        printf("]\n");
    }
}

/******************************************************************************
 * CC Testing Routines
 *****************************************************************************/

/**
 * @brief CPU-based reference CC algorithm using Boost Graph Library
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in]  graph  Reference to the CSR graph we process on
 * @param[out] labels Host-side vector to store the component id for each node in the graph
 * @param[in] quiet Don't print out anything to stdout
 *
 * \return Number of connected components in the graph
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
unsigned int ReferenceCC(
    const Csr<VertexId, Value, SizeT> &graph,
    int *labels,
    bool quiet = false)
{
    using namespace boost;
    SizeT    *row_offsets    = graph.row_offsets;
    VertexId *column_indices = graph.column_indices;
    SizeT     num_nodes      = graph.nodes;

    typedef adjacency_list <vecS, vecS, undirectedS> Graph;
    Graph G;
    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j)
        {
            add_edge(i, column_indices[j], G);
        }
    }
    CpuTimer cpu_timer;
    cpu_timer.Start();
    int num_components = connected_components(G, &labels[0]);
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    if (!quiet) { printf("CPU CC finished in %lf msec.\n", elapsed); }
    return num_components;
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
 * @tparam MARK_PREDECESSORS
 * @tpatam ENABLE_IDEMPOTENCE
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source_path Host-side vector to store CPU computed labels for each node
 * @param[in] predecessor Host-side vector to store CPU computed predecessor for each node
 * @param[in] src Source node where BFS starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void ReferenceBFS(
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

/******************************************************************************
* BC Testing Routines
*****************************************************************************/

/**
 * @brief Graph edge properties (bundled properties)
 */
struct EdgeProperties
{
    int weight;
};

/**
 * @brief A simple CPU-based reference BC ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to graph we process on
 * @param[in] bc_values Pointer to node bc value
 * @param[in] ebc_values Pointer to edge bc value
 * @param[in] sigmas Pointer to node sigma value
 * @param[in] src VertexId of source node if there is any
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void ReferenceBC(
    const Csr<VertexId, Value, SizeT> &graph,
    Value                             *bc_values,
    Value                             *ebc_values,
    Value                             *sigmas,
    VertexId                          *source_path,
    VertexId                           src,
    bool                               quiet = false)
{
    typedef Coo<VertexId, Value> EdgeTupleType;
    EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * graph.edges);
    if (src == -1)
    {
        // Perform full exact BC using BGL

        using namespace boost;
        typedef adjacency_list <setS, vecS, undirectedS, no_property,
                EdgeProperties> Graph;
        typedef Graph::vertex_descriptor Vertex;
        typedef Graph::edge_descriptor Edge;

        Graph G;
        for (int i = 0; i < graph.nodes; ++i)
        {
            for (int j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
            {
                add_edge(vertex(i, G), vertex(graph.column_indices[j], G), G);
            }
        }

        typedef std::map<Edge, int> StdEdgeIndexMap;
        StdEdgeIndexMap my_e_index;
        typedef boost::associative_property_map< StdEdgeIndexMap > EdgeIndexMap;
        EdgeIndexMap e_index(my_e_index);

        int i = 0;
        BGL_FORALL_EDGES(edge, G, Graph)
        {
            my_e_index.insert(std::pair<Edge, int>(edge, i));
            ++i;
        }

        // Define EdgeCentralityMap
        std::vector< double > e_centrality_vec(boost::num_edges(G), 0.0);
        // Create the external property map
        boost::iterator_property_map< std::vector< double >::iterator,
              EdgeIndexMap >
              e_centrality_map(e_centrality_vec.begin(), e_index);

        // Define VertexCentralityMap
        typedef boost::property_map< Graph, boost::vertex_index_t>::type
        VertexIndexMap;
        VertexIndexMap v_index = get(boost::vertex_index, G);
        std::vector< double > v_centrality_vec(boost::num_vertices(G), 0.0);

        // Create the external property map
        boost::iterator_property_map< std::vector< double >::iterator,
              VertexIndexMap>
              v_centrality_map(v_centrality_vec.begin(), v_index);

        // Perform BC
        CpuTimer cpu_timer;
        cpu_timer.Start();
        brandes_betweenness_centrality(G, v_centrality_map, e_centrality_map);
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();

        BGL_FORALL_VERTICES(vertex, G, Graph)
        {
            bc_values[vertex] = (Value)v_centrality_map[vertex];
        }

        int idx = 0;
        BGL_FORALL_EDGES(edge, G, Graph)
        {
            coo[idx].row = source(edge, G);
            coo[idx].col = target(edge, G);
            coo[idx++].val = (Value)e_centrality_map[edge];
            coo[idx].col = source(edge, G);
            coo[idx].row = target(edge, G);
            coo[idx++].val = (Value)e_centrality_map[edge];
        }

        std::stable_sort(coo, coo + graph.edges,
                         RowFirstTupleCompare<EdgeTupleType>);

        for (idx = 0; idx < graph.edges; ++idx)
        {
            //std::cout << coo[idx].row << "," << coo[idx].col
            //          << ":" << coo[idx].val << std::endl;
            //ebc_values[idx] = coo[idx].val;
        }

        if (!quiet)
        {
            printf("CPU BC finished in %lf msec.", elapsed);
        }
    }
    else
    {
        //Simple BFS pass to get single pass BC
        //VertexId *source_path = new VertexId[graph.nodes];

        //initialize distances
        for (VertexId i = 0; i < graph.nodes; ++i)
        {
            source_path[i] = -1;
            bc_values[i] = 0;
            sigmas[i] = 0;
        }
        source_path[src] = 0;
        VertexId search_depth = 0;
        sigmas[src] = 1;

        // Initialize queue for managing previously-discovered nodes
        std::deque<VertexId> frontier;
        frontier.push_back(src);

        //
        //Perform one pass of BFS for one source
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
            int edges_begin = graph.row_offsets[dequeued_node];
            int edges_end = graph.row_offsets[dequeued_node + 1];

            for (int edge = edges_begin; edge < edges_end; ++edge)
            {
                // Lookup neighbor and enqueue if undiscovered
                VertexId neighbor = graph.column_indices[edge];
                if (source_path[neighbor] == -1)
                {
                    source_path[neighbor] = neighbor_dist;
                    sigmas[neighbor] += sigmas[dequeued_node];
                    if (search_depth < neighbor_dist)
                    {
                        search_depth = neighbor_dist;
                    }

                    frontier.push_back(neighbor);
                }
                else
                {
                    if (source_path[neighbor] == source_path[dequeued_node] + 1)
                        sigmas[neighbor] += sigmas[dequeued_node];
                }
            }
        }
        search_depth++;

        for (int iter = search_depth - 2; iter > 0; --iter)
        {

            int cur_level = 0;
            for (int node = 0; node < graph.nodes; ++node)
            {
                if (source_path[node] == iter)
                {
                    ++cur_level;
                    int edges_begin = graph.row_offsets[node];
                    int edges_end = graph.row_offsets[node + 1];

                    for (int edge = edges_begin; edge < edges_end; ++edge)
                    {
                        VertexId neighbor = graph.column_indices[edge];
                        if (source_path[neighbor] == iter + 1)
                        {
                            bc_values[node] +=
                                1.0f * sigmas[node] / sigmas[neighbor] *
                                (1.0f + bc_values[neighbor]);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < graph.nodes; ++i)
        {
            bc_values[i] *= 0.5f;
        }

        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();

        if (!quiet)
        {
            printf("CPU BC finished in %lf msec. Search depth: %d\n",
                   elapsed, search_depth);
        }

        //delete[] source_path;
    }
    free(coo);
}

/**
 * @brief Convert component IDs.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] labels
 * @param[in] num_nodes
 * @param[in] num_components
 */
template <
    typename VertexId,
    typename SizeT >
void ConvertIDs(
    VertexId *labels,
    SizeT    num_nodes,
    SizeT    num_components)
{
    VertexId *min_nodes = new VertexId[num_nodes];

    for (int cc = 0; cc < num_nodes; cc++)
        min_nodes[cc] = num_nodes;
    for (int node = 0; node < num_nodes; node++)
        if (min_nodes[labels[node]] > node) min_nodes[labels[node]] = node;
    for (int node = 0; node < num_nodes; node++)
        labels[node] = min_nodes[labels[node]];
    delete[] min_nodes; min_nodes = NULL;
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
VertexId RunCC(Info<VertexId, Value, SizeT> *info)
{
    typedef CCProblem < VertexId,
            SizeT,
            Value,
            false > CcProblem;  // use double buffer for advance and filter

    typedef CCEnactor < CcProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK > CcEnactor;

    // parse configurations from mObject info
    Csr<VertexId, Value, SizeT> *graph = info->csr_ptr;
    int max_grid_size            = info->info["max_grid_size"].get_int();
    int num_gpus                 = info->info["num_gpus"].get_int();
    double max_queue_sizing      = info->info["max_queue_sizing"].get_real();
    double max_queue_sizing1     = info->info["max_queue_sizing1"].get_real();
    double max_in_sizing         = info->info["max_in_sizing"].get_real();
    std::string partition_method = info->info["partition_method"].get_str();
    double partition_factor      = info->info["partition_factor"].get_real();
    int partition_seed           = info->info["partition_seed"].get_int();
    bool quiet_mode              = info->info["quiet_mode"].get_bool();
    bool quick_mode              = info->info["quick_mode"].get_bool();
    bool stream_from_host        = info->info["stream_from_host"].get_bool();
    int traversal_mode           = info->info["traversal_mode"].get_int();
    int iterations               = 1; //set to 1 for now. info->info["num_iteration"].get_int();

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side array (for both reference and GPU-computed results)
    VertexId    *reference_component_ids = new VertexId[graph->nodes];
    VertexId    *h_component_ids        = new VertexId[graph->nodes];
    VertexId    *reference_check        = (quick_mode) ? NULL : reference_component_ids;
    unsigned int ref_num_components     = 0;

    //printf("0: node %d: %d -> %d, node %d: %d -> %d\n", 131070, graph->row_offsets[131070], graph->row_offsets[131071], 131071, graph->row_offsets[131071], graph->row_offsets[131072]);
    //for (int edge = 0; edge < graph->edges; edge ++)
    //{
    //    if (graph->column_indices[edge] == 131070 || graph->column_indices[edge] == 131071)
    //    printf("edge %d: -> %d\n", edge, graph->column_indices[edge]);
    //}

    //util::cpu_mt::PrintCPUArray("row_offsets", graph->row_offsets, graph->nodes+1);
    //util::cpu_mt::PrintCPUArray("colunm_indices", graph->column_indices, graph->edges);
    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    CcEnactor* enactor = new CcEnactor(num_gpus, gpu_idx);  // enactor map
    CcProblem* problem = new CcProblem;  // allocate problem on GPU

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
                  "CC Problem Initialization Failed", __FILE__, __LINE__);
    util::GRError(enactor->Init(context, problem, max_grid_size),
                  "CC Enactor Init failed", __FILE__, __LINE__);

    // compute reference CPU CC
    if (reference_check != NULL)
    {
        if (!quiet_mode) { printf("Computing reference value ...\n"); }
        ref_num_components = ReferenceCC(*graph, reference_check, quiet_mode);
        if (!quiet_mode) { printf("\n"); }
    }

    // perform CC
    CpuTimer cpu_timer;
    float elapsed = 0.0f;

    for (SizeT iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(
                          enactor->GetFrontierType(), max_queue_sizing),
                      "CC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(enactor->Reset(),
                      "CC Enactor Reset failed", __FILE__, __LINE__);

        if (!quiet_mode)
        {
            printf("_________________________\n"); fflush(stdout);
        }
        cpu_timer.Start();
        util::GRError(enactor->Enact(),
                      "CC Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        if (!quiet_mode)
        {
            printf("-------------------------\n"); fflush(stdout);
        }
        elapsed += cpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    // copy out results
    util::GRError(problem->Extract(h_component_ids),
                  "CC Problem Data Extraction Failed", __FILE__, __LINE__);

    // validity
    if (reference_check != NULL)
    {
        if (ref_num_components == problem->num_components)
        {
            if (!quiet_mode)
            {
                printf("CORRECT. Component Count: %d\n", ref_num_components);
            }
        }
        else
        {
            if (!quiet_mode)
            {
                printf(
                    "INCORRECT. Ref Component Count: %d, "
                    "GPU Computed Component Count: %d\n",
                    ref_num_components, problem->num_components);
            }
        }
    }
    else
    {
        if (!quiet_mode)
        {
            printf("Component Count: %lld\n", (long long) problem->num_components);
        }
    }
    if (reference_check != NULL)
    {
        ConvertIDs<VertexId, SizeT>(reference_check, graph->nodes, ref_num_components);
        ConvertIDs<VertexId, SizeT>(h_component_ids, graph->nodes, problem->num_components);
        if (!quiet_mode)
        {
            printf("Label Validity: ");
        }
        int error_num = CompareResults(
                            h_component_ids, reference_check, graph->nodes, true, quiet_mode);
        if (error_num > 0)
        {
            if (!quiet_mode) { printf("%d errors occurred.\n", error_num); }
        }
        else
        {
            if (!quiet_mode) { printf("\n"); }
        }
    }

        // Compute size and root of each component
        VertexId     *h_roots      = new VertexId    [problem->num_components];
        unsigned int *h_histograms = new unsigned int[problem->num_components];

        //printf("num_components = %d\n", problem->num_components);
        problem->ComputeCCHistogram(h_component_ids, h_roots, h_histograms);
        //printf("num_components = %d\n", problem->num_components);

            // Display Solution
        VertexId largest_cc_id = DisplayCCSolution(h_component_ids, graph->nodes,
                            problem->num_components, h_roots, h_histograms);

        if (h_roots     ) {delete[] h_roots     ; h_roots      = NULL;}
        if (h_histograms) {delete[] h_histograms; h_histograms = NULL;}

    info->ComputeCommonStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), elapsed, h_component_ids);

    if (!quiet_mode)
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject

    if (!quiet_mode)
    {
        printf("\n\tMemory Usage(B)\t");
        for (int gpu = 0; gpu < num_gpus; gpu++)
            if (num_gpus > 1)
            {
                if (gpu != 0) printf(" #keys%d\t #ins%d,0\t #ins%d,1", gpu, gpu, gpu);
                else printf(" $keys%d", gpu);
            }
            else printf(" #keys%d", gpu);
        if (num_gpus > 1) printf(" #keys%d", num_gpus);
        printf("\n");

        double max_key_sizing = 0, max_in_sizing_ = 0;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            size_t gpu_free, dummy;
            cudaSetDevice(gpu_idx[gpu]);
            cudaMemGetInfo(&gpu_free, &dummy);
            printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
            for (int i = 0; i < num_gpus; i++)
            {
                SizeT x = problem->data_slices[gpu]->frontier_queues[i].keys[0].GetSize();
                printf("\t %d", x);
                double factor = 1.0 * x / (num_gpus > 1 ? problem->graph_slices[gpu]->in_counter[i] : problem->graph_slices[gpu]->nodes);
                if (factor > max_key_sizing) max_key_sizing = factor;
                if (num_gpus > 1 && i != 0 )
                    for (int t = 0; t < 2; t++)
                    {
                        x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                        printf("\t %d", x);
                        factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
                        if (factor > max_in_sizing_) max_in_sizing_ = factor;
                    }
            }
            if (num_gpus > 1) printf("\t %d", problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize());
            printf("\n");
        }
        printf("\t key_sizing =\t %lf", max_key_sizing);
        if (num_gpus > 1) printf("\t in_sizing =\t %lf", max_in_sizing_);
        printf("\n");
    }

    // Cleanup
    if (org_size               ) {delete[] org_size               ; org_size                = NULL;}
    if (problem                ) {delete   problem                ; problem                 = NULL;}
    if (enactor                ) {delete   enactor                ; enactor                 = NULL;}
    if (reference_component_ids) {delete[] reference_component_ids; reference_component_ids = NULL;}
    if (h_component_ids        ) {delete[] h_component_ids        ; h_component_ids         = NULL;}

    return largest_cc_id;
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
VertexId RunCC_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
    {
        return RunCC<VertexId, Value, SizeT, INSTRUMENT, DEBUG,  true>(info);
    }
    else
    {
        return RunCC<VertexId, Value, SizeT, INSTRUMENT, DEBUG, false>(info);
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
VertexId RunCC_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
    {
        return RunCC_size_check<VertexId, Value, SizeT, INSTRUMENT,  true>(info);
    }
    else
    {
        return RunCC_size_check<VertexId, Value, SizeT, INSTRUMENT, false>(info);
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
VertexId RunCC_instrumented(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["instrument"].get_bool())
    {
        return RunCC_debug<VertexId, Value, SizeT, true>(info);
    }
    else
    {
        return RunCC_debug<VertexId, Value, SizeT, false>(info);
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
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK,
    bool        MARK_PREDECESSORS,
    bool        ENABLE_IDEMPOTENCE >
void RunBFS(Info<VertexId, Value, SizeT> *info, VertexId src_input)
{
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
    Csr<VertexId, Value, SizeT> *graph = info->csr_ptr;
    VertexId src                 = src_input;
    info->info["source_vertex"] = src;
    int max_grid_size            = info->info["max_grid_size"].get_int();
    int num_gpus                 = info->info["num_gpus"].get_int();
    double max_queue_sizing      = info->info["max_queue_sizing"].get_real();
    double max_queue_sizing1     = info->info["max_queue_sizing1"].get_real();
    double max_in_sizing         = info->info["max_in_sizing"].get_real();
    std::string partition_method = info->info["partition_method"].get_str();
    double partition_factor      = info->info["partition_factor"].get_real();
    int partition_seed           = info->info["partition_seed"].get_int();
    bool quiet_mode              = info->info["quiet_mode"].get_bool();
    bool quick_mode              = info->info["quick_mode"].get_bool();
    bool stream_from_host        = info->info["stream_from_host"].get_bool();
    int traversal_mode           = info->info["traversal_mode"].get_int();
    int iterations               = 1; //disable since doesn't support mgpu stop condition. info->info["num_iteration"].get_int();

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // allocate host-side label array (for both reference and GPU results)
    VertexId *reference_labels      = new VertexId[graph->nodes];
    VertexId *reference_preds       = new VertexId[graph->nodes];
    VertexId *h_labels              = new VertexId[graph->nodes];
    VertexId *reference_check_label = (quick_mode) ? NULL : reference_labels;
    VertexId *reference_check_preds = NULL;
    VertexId *h_preds               = NULL;

    if (MARK_PREDECESSORS)
    {
        h_preds = new VertexId[graph->nodes];
        if (!quick_mode)
        {
            reference_check_preds = reference_preds;
        }
    }

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    BfsEnactor* enactor = new BfsEnactor(num_gpus, gpu_idx);  // enactor map
    BfsProblem* problem = new BfsProblem;  // allocate problem on GPU

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
                  "BFS Problem Init failed", __FILE__, __LINE__);

    util::GRError(enactor->Init(
                      context, problem, max_grid_size, traversal_mode),
                  "BFS Enactor Init failed", __FILE__, __LINE__);

    // compute reference CPU BFS solution for source-distance
    if (reference_check_label != NULL)
    {
        if (!quiet_mode)
        {
            printf("Computing reference value ...\n");
        }
        ReferenceBFS<VertexId, Value, SizeT,
                     MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>(
                         graph, reference_check_label,
                         reference_check_preds, src, quiet_mode);
        if (!quiet_mode)
        {
            printf("\n");
        }
    }

    // perform BFS
    double elapsed = 0.0f;
    CpuTimer cpu_timer;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(
                          src, enactor->GetFrontierType(),
                          max_queue_sizing, max_queue_sizing1),
                      "BFS Problem Data Reset Failed", __FILE__, __LINE__);

        util::GRError(enactor->Reset(),
                      "BFS Enactor Reset failed", __FILE__, __LINE__);

        util::GRError("Error before Enact", __FILE__, __LINE__);

        if (!quiet_mode)
        {
            printf("__________________________\n"); fflush(stdout);
        }

        cpu_timer.Start();
        util::GRError(enactor->Enact(src, traversal_mode),
                      "BFS Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();

        if (!quiet_mode)
        {
            printf("--------------------------\n"); fflush(stdout);
        }

        elapsed += cpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    // copy out results
    util::GRError(problem->Extract(h_labels, h_preds),
                  "BFS Problem Data Extraction Failed", __FILE__, __LINE__);

    // verify the result
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

    // display Solution
    if (!quiet_mode)
    {
        DisplayBFSSolution<VertexId, SizeT, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>
        (h_labels, h_preds, graph->nodes, quiet_mode);
    }

    info->ComputeTraversalStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), elapsed, h_labels);

    if (!quiet_mode)
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject

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
void RunBFS_enable_idempotence(Info<VertexId, Value, SizeT> *info, VertexId src)
{
    if (info->info["idempotent"].get_bool())
    {
        RunBFS <VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
                 MARK_PREDECESSORS, true > (info, src);
    }
    else
    {
        RunBFS <VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
                 MARK_PREDECESSORS, false> (info, src);
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
void RunBFS_mark_predecessors(Info<VertexId, Value, SizeT> *info, VertexId src)
{
    if (info->info["mark_predecessors"].get_bool())
    {
        RunBFS_enable_idempotence<VertexId, Value, SizeT, INSTRUMENT,
                                    DEBUG, SIZE_CHECK,  true> (info, src);
    }
    else
    {
        RunBFS_enable_idempotence<VertexId, Value, SizeT, INSTRUMENT,
                                    DEBUG, SIZE_CHECK, false> (info, src);
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
void RunBFS_size_check(Info<VertexId, Value, SizeT> *info, VertexId src)
{
    if (info->info["size_check"].get_bool())
    {
        RunBFS_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT,
                                   DEBUG,  true>(info, src);
    }
    else
    {
        RunBFS_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT,
                                   DEBUG, false>(info, src);
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
void RunBFS_debug(Info<VertexId, Value, SizeT> *info, VertexId src)
{
    if (info->info["debug_mode"].get_bool())
    {
        RunBFS_size_check<VertexId, Value, SizeT, INSTRUMENT,  true>(info, src);
    }
    else
    {
        RunBFS_size_check<VertexId, Value, SizeT, INSTRUMENT, false>(info, src);
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
void RunBFS_instrumented(Info<VertexId, Value, SizeT> *info, VertexId src)
{
    if (info->info["instrument"].get_bool())
    {
        RunBFS_debug<VertexId, Value, SizeT, true>(info, src);
    }
    else
    {
        RunBFS_debug<VertexId, Value, SizeT, false>(info, src);
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
void RunBC(Info<VertexId, Value, SizeT> *info)
{
    typedef BCProblem < VertexId,
            SizeT,
            Value,
            true,   // MARK_PREDECESSORS
            false > BcProblem;  //does not use double buffer

    typedef BCEnactor < BcProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK >
            BcEnactor;

    // parse configurations from mObject info
    Csr<VertexId, Value, SizeT> *graph = info->csr_ptr;
    VertexId src                 = info->info["source_vertex"].get_int64();
    bool quiet_mode              = info->info["quiet_mode"].get_bool();
    int max_grid_size            = info->info["max_grid_size"].get_int();
    int num_gpus                 = info->info["num_gpus"].get_int();
    double max_queue_sizing      = info->info["max_queue_sizing"].get_real();
    double max_queue_sizing1     = info->info["max_queue_sizing1"].get_real();
    double max_in_sizing         = info->info["max_in_sizing"].get_real();
    std::string partition_method = info->info["partition_method"].get_str();
    double partition_factor      = info->info["partition_factor"].get_real();
    int partition_seed           = info->info["partition_seed"].get_int();
    bool quick_mode              = info->info["quick_mode"].get_bool();
    bool stream_from_host        = info->info["stream_from_host"].get_bool();
    int iterations               = 1; // force to 1 info->info["num_iteration"].get_int();
    std::string ref_filename     = info->info["ref_filename"].get_str();

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side array (for both reference and GPU-computed results)
    Value        *reference_bc_values        = new Value   [graph->nodes];
    Value        *reference_ebc_values       = new Value   [graph->edges];
    Value        *reference_sigmas           = new Value   [graph->nodes];
    VertexId     *reference_labels           = new VertexId[graph->nodes];
    Value        *h_sigmas                   = new Value   [graph->nodes];
    Value        *h_bc_values                = new Value   [graph->nodes];
    Value        *h_ebc_values               = new Value   [graph->edges];
    VertexId     *h_labels                   = new VertexId[graph->nodes];
    Value        *reference_check_bc_values  = (quick_mode)                ? NULL : reference_bc_values;
    Value        *reference_check_ebc_values = (quick_mode || (src != -1)) ? NULL : reference_ebc_values;
    Value        *reference_check_sigmas     = (quick_mode || (src == -1)) ? NULL : reference_sigmas;
    VertexId     *reference_check_labels     = (quick_mode || (src == -1)) ? NULL : reference_labels;

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    BcEnactor* enactor = new BcEnactor(num_gpus, gpu_idx);  // enactor map
    BcProblem* problem = new BcProblem;  // allocate problem on GPU

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
                  "BC Problem Initialization Failed", __FILE__, __LINE__);

    util::GRError(enactor->Init(context, problem, max_grid_size),
                  "BC Enactor init failed", __FILE__, __LINE__);

    // compute reference CPU BC solution for source-distance
    if (reference_check_bc_values != NULL)
    {
        if (ref_filename.empty())
        {
            if (!quiet_mode) { printf("Computing reference value ...\n"); }
            ReferenceBC(
                *graph,
                reference_check_bc_values,
                reference_check_ebc_values,
                reference_check_sigmas,
                reference_check_labels,
                src,
                quiet_mode);
            if (!quiet_mode) { printf("\n"); }
        }
        else
        {
            std::ifstream fin;
            fin.open(ref_filename.c_str(), std::ios::binary);
            for (int i = 0; i < graph->nodes; ++i)
            {
                fin.read(reinterpret_cast<char*>(&reference_check_bc_values[i]), sizeof(Value));
            }
            fin.close();
        }
    }

    // perform BC
    double elapsed  = 0.0f;
    CpuTimer cpu_timer;

    VertexId start_src, end_src;
    if (src == -1)
    {
        start_src = 0;
        end_src = graph->nodes;
    }
    else
    {
        start_src = src;
        end_src = src + 1;
    }

    for (int iter = 0; iter < iterations; ++iter)
    {
        printf("iteration:%d\n", iter);
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            util::SetDevice(gpu_idx[gpu]);
            util::MemsetKernel <<< 128, 128>>>(
                problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE),
                (Value)0.0f, (int)(problem->sub_graphs[gpu].nodes));
        }
        util::GRError(problem->Reset(
                          0, enactor->GetFrontierType(),
                          max_queue_sizing, max_queue_sizing1),
                      "BC Problem Data Reset Failed", __FILE__, __LINE__);

        if (!quiet_mode)
        {
            printf("__________________________\n"); fflush(stdout);
        }
        cpu_timer.Start();
        for (VertexId i = start_src; i < end_src; ++i)
        {
            util::GRError(problem->Reset(
                              i, enactor->GetFrontierType(),
                              max_queue_sizing, max_queue_sizing1),
                          "BC Problem Data Reset Failed", __FILE__, __LINE__);
            util::GRError(enactor ->Reset(),
                          "BC Enactor Reset failed", __FILE__, __LINE__);
            util::GRError(enactor ->Enact(i),
                          "BC Problem Enact Failed", __FILE__, __LINE__);
        }
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            util::SetDevice(gpu_idx[gpu]);
            util::MemsetScaleKernel <<< 128, 128>>>(
                problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE),
                (Value)0.5f, (int)(problem->sub_graphs[gpu].nodes));
        }
        cpu_timer.Stop();
        if (!quiet_mode)
        {
            printf("--------------------------\n"); fflush(stdout);
        }
        elapsed += cpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    // Copy out results
    util::GRError(problem->Extract(
                      h_sigmas, h_bc_values, h_ebc_values, h_labels),
                  "BC Problem Data Extraction Failed", __FILE__, __LINE__);

    // Verify the result
    if (reference_check_bc_values != NULL)
    {
        //util::cpu_mt::PrintCPUArray<SizeT, Value>("reference_check_bc_values", reference_check_bc_values, graph->nodes);
        //util::cpu_mt::PrintCPUArray<SizeT, Value>("bc_values", h_bc_values, graph->nodes);
        if (!quiet_mode) { printf("Validity BC Value: "); }
        int num_error = CompareResults(
                            h_bc_values, reference_check_bc_values,
                            graph->nodes, true, quiet_mode);
        if (num_error > 0)
        {
            if (!quiet_mode) { printf("Number of errors occurred: %d\n", num_error); }
        }
        if (!quiet_mode) { printf("\n"); }
    }
    if (reference_check_ebc_values != NULL)
    {
        if (!quiet_mode) { printf("Validity Edge BC Value: "); }
        int num_error = CompareResults(
                            h_ebc_values, reference_check_ebc_values,
                            graph->edges, true, quiet_mode);
        if (num_error > 0)
        {
            if (!quiet_mode) { printf("Number of errors occurred: %d\n", num_error); }
        }
        if (!quiet_mode) { printf("\n"); }
    }
    if (reference_check_sigmas != NULL)
    {
        if (!quiet_mode) { printf("Validity Sigma: "); }
        int num_error = CompareResults(
                            h_sigmas, reference_check_sigmas,
                            graph->nodes, true, quiet_mode);
        if (num_error > 0)
        {
            if (!quiet_mode)
            {
                printf("Number of errors occurred: %d\n", num_error);
            }
        }
        if (!quiet_mode) { printf("\n"); }
    }
    if (reference_check_labels != NULL)
    {
        if (!quiet_mode) { printf("Validity labels: "); }
        int num_error = CompareResults(
                            h_labels, reference_check_labels,
                            graph->nodes, true, quiet_mode);
        if (num_error > 0)
        {
            if (!quiet_mode)
            {
                printf("Number of errors occurred: %d\n", num_error);
            }
        }
        if (!quiet_mode) { printf("\n"); }
    }

    if (!quiet_mode)
    {
        // Display Solution
        DisplayBCSolution(h_sigmas, h_bc_values, graph->nodes);
    }

    info->ComputeTraversalStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), elapsed, h_labels);

    if (!quiet_mode)
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject

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

    // Cleanup
    if (org_size            ) {delete[] org_size            ; org_size             = NULL;}
    if (problem             ) {delete   problem             ; problem              = NULL;}
    if (enactor             ) {delete   enactor             ; enactor              = NULL;}
    if (reference_sigmas    ) {delete[] reference_sigmas    ; reference_sigmas     = NULL;}
    if (reference_bc_values ) {delete[] reference_bc_values ; reference_bc_values  = NULL;}
    if (reference_ebc_values) {delete[] reference_ebc_values; reference_ebc_values = NULL;}
    if (reference_labels    ) {delete[] reference_labels    ; reference_labels     = NULL;}
    if (h_sigmas            ) {delete[] h_sigmas            ; h_sigmas             = NULL;}
    if (h_bc_values         ) {delete[] h_bc_values         ; h_bc_values          = NULL;}
    if (h_ebc_values        ) {delete[] h_ebc_values        ; h_ebc_values         = NULL;}
    if (h_labels            ) {delete[] h_labels            ; h_labels             = NULL;}
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
void RunBC_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
    {
        RunBC<VertexId, Value, SizeT, INSTRUMENT, DEBUG,  true>(info);
    }
    else
    {
        RunBC<VertexId, Value, SizeT, INSTRUMENT, DEBUG, false>(info);
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
void RunBC_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
    {
        RunBC_size_check<VertexId, Value, SizeT, INSTRUMENT,  true>(info);
    }
    else
    {
        RunBC_size_check<VertexId, Value, SizeT, INSTRUMENT, false>(info);
    }
}

/**
 * @brief Test entry
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
void RunBC_instrumented(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["instrument"].get_bool())
    {
        RunBC_debug<VertexId, Value, SizeT,  true>(info);
    }
    else
    {
        RunBC_debug<VertexId, Value, SizeT, false>(info);
    }
}

/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
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

    Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
    Info<VertexId, Value, SizeT> *info_cc = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info_cc->info["undirected"] = true;   // require undirected input graph

    info_cc->Init("CC", args, csr);  // initialize Info structure
    graphio::RemoveStandaloneNodes<VertexId, Value, SizeT>(
        &csr, args.CheckCmdLineFlag("quiet"));
    VertexId src = RunCC_instrumented<VertexId, Value, SizeT>(info_cc);  // run test

    Info<VertexId, Value, SizeT> *info_bfs = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info_bfs->info["undirected"] = args.CheckCmdLineFlag("undirected");

    info_bfs->Init("BFS", args, csr);  // initialize Info structure
    RunBFS_instrumented<VertexId, Value, SizeT>(info_bfs, src);  // run test

    typedef float FloatValue;   // Use float as the value type

    Csr<VertexId, FloatValue, SizeT> csr_bc(false);  // graph we process on
    Info<VertexId, FloatValue, SizeT> *info_bc = new Info<VertexId, FloatValue, SizeT>;

    // graph construction or generation related parameters
    info_bc->info["undirected"] = true;  // require undirected input graph

    info_bc->Init("BC", args, csr_bc);  // initialize Info structure
    RunBC_instrumented<VertexId, FloatValue, SizeT>(info_bc);  // run test

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:




