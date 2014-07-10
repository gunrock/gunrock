// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bc.cu
 *
 * @brief Simple test driver program for BC.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BC includes
#include <gunrock/app/bc/bc_enactor.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bc;


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
    printf("\ntest_bc <graph type> <graph type args> [--device=<device_index>] "
           "[--instrumented] [--src=<source index>] [--quick] [--v]"
           "[--queue-sizing=<scale factor>] [--ref-file=<reference filename>]\n"
           "\n"
           "Graph types and args:\n"
           "  market [<file>]\n"
           "    Reads a Matrix-Market coordinate-formatted graph of undirected\n"
           "    edges from stdin (or from the optionally-specified file).\n"
           "--device=<device_index>: Set GPU device for running the graph primitive.\n"
           "--undirected: If set then treat the graph as undirected graph.\n"
           "--instrumented: If set then kernels keep track of queue-search_depth\n"
           "and barrier duty (a relative indicator of load imbalance.)\n"
           "--src=<source index>: When source index is -1, compute BC value for each\n"
           "node. Otherwise, debug the delta value for one node\n"
           "--quick: If set will skip the CPU validation code.\n"
           "--queue-sizing Allocates a frontier queue sized at (graph-edges * <scale factor>).\n"
           "Default is 1.0.\n"
           "--v: If set, enable verbose output, keep track of the kernel running.\n"
           "--ref-file: If set, use pre-computed result stored in ref-file to verify.\n"
           );
}

/**
 * @brief Displays the BC result (sigma value and BC value)
 *
 * @param[in] sigmas
 * @param[in] bc_values
 * @param[in] nodes
 */
template<typename Value, typename SizeT>
void DisplaySolution(Value *sigmas, Value *bc_values, SizeT nodes)
{
    if (nodes < 40) {
        printf("[");
        for (SizeT i = 0; i < nodes; ++i) {
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
 * @param[in] graph Reference to ...
 * @param[in] bc_values Pointer to ...
 * @param[in] sigmas Pointer to ...
 * @param[in] src VertexId of ...
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void RefCPUBC(
    const Csr<VertexId, Value, SizeT>       &graph,
    Value                                   *bc_values,
    Value                                   *ebc_values,
    Value                                   *sigmas,
    VertexId                                src)
{
    typedef Coo<VertexId, Value> EdgeTupleType;
    EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * graph.edges);
    if (src == -1) {
        // Perform full exact BC using BGL

        using namespace boost;
        typedef adjacency_list <setS, vecS, undirectedS, no_property,
                                EdgeProperties> Graph;
        typedef Graph::vertex_descriptor Vertex;
        typedef Graph::edge_descriptor Edge;

        Graph G;
        for (int i = 0; i < graph.nodes; ++i)
        {
            for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
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

        //
        // Perform BC
        //
        CpuTimer cpu_timer;
        cpu_timer.Start();
        brandes_betweenness_centrality( G, v_centrality_map, e_centrality_map );
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

        std::stable_sort(coo, coo+graph.edges, RowFirstTupleCompare<EdgeTupleType>);

        for (idx = 0; idx < graph.edges; ++idx) {
            //std::cout << coo[idx].row << "," << coo[idx].col << ":" << coo[idx].val << std::endl;
            ebc_values[idx] = coo[idx].val;
        }

        printf("CPU BC finished in %lf msec.", elapsed);
    }
    else {
        //Simple BFS pass to get single pass BC
        VertexId *source_path = new VertexId[graph.nodes];

        //initialize distances
        for (VertexId i = 0; i < graph.nodes; ++i) {
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
        while (!frontier.empty()) {

            // Dequeue node from frontier
            VertexId dequeued_node = frontier.front();
            frontier.pop_front();
            VertexId neighbor_dist = source_path[dequeued_node] + 1;

            // Locate adjacency list
            int edges_begin = graph.row_offsets[dequeued_node];
            int edges_end = graph.row_offsets[dequeued_node + 1];

            for (int edge = edges_begin; edge < edges_end; ++edge) {
                // Lookup neighbor and enqueue if undiscovered
                VertexId neighbor = graph.column_indices[edge];
                if (source_path[neighbor] == -1) {
                    source_path[neighbor] = neighbor_dist;
                    sigmas[neighbor] += sigmas[dequeued_node];
                    if (search_depth < neighbor_dist) {
                        search_depth = neighbor_dist;
                    }

                    frontier.push_back(neighbor);
                }
                else {
                    if (source_path[neighbor] == source_path[dequeued_node]+1)
                        sigmas[neighbor] += sigmas[dequeued_node];
                }
            }
        }
        search_depth++;

        for (int iter = search_depth - 2; iter > 0; --iter)
        {
            for (int node = 0; node < graph.nodes; ++node)
            {
                if (source_path[node] == iter) {
                    int edges_begin = graph.row_offsets[node];
                    int edges_end = graph.row_offsets[node+1];

                    for (int edge = edges_begin; edge < edges_end; ++edge) {
                        VertexId neighbor = graph.column_indices[edge];
                        if (source_path[neighbor] == iter + 1) {
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

        printf("CPU BFS finished in %lf msec. Search depth is:%d\n",
               elapsed, search_depth);

        delete[] source_path;
    }
    free(coo);
}

/**
 * @brief Run betweenness centrality tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph object defined in main driver
 * @param[in] src
 * @param[in] ref_filename 
 * @param[in] max_grid_size
 * @param[in] num_gpus
 * @param[in] max_queue_sizing
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    VertexId    src,
    std::string &ref_filename,
    int         max_grid_size,
    int         num_gpus,
    double      max_queue_sizing,
    ContextPtr  *context,
    std::string partition_method,
    int         *gpu_idx)
{
    typedef BCProblem<
        VertexId,
        SizeT,
        Value,
        true,   // MARK_PREDECESSORS
        false> Problem; //does not use double buffer

    // Allocate host-side array (for both reference and gpu-computed results)
    Value *reference_bc_values        = new Value[graph.nodes];
    Value *reference_ebc_values       = new Value[graph.edges];
    Value *reference_sigmas           = new Value[graph.nodes];
    Value *h_sigmas                   = new Value[graph.nodes];
    Value *h_bc_values                = new Value[graph.nodes];
    Value *h_ebc_values               = new Value[graph.edges];
    Value *reference_check_bc_values  = (g_quick)                ? NULL : reference_bc_values;
    Value *reference_check_ebc_values = (g_quick || (src != -1)) ? NULL : reference_ebc_values;
    Value *reference_check_sigmas     = (g_quick || (src == -1)) ? NULL : reference_sigmas;

    // Allocate BC enactor map
    BCEnactor<Problem, INSTRUMENT>* bc_enactor
        = new BCEnactor<Problem, INSTRUMENT>(g_verbose, num_gpus, gpu_idx);
    
    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
            g_stream_from_host,
            graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method), "BC Problem Initialization Failed", __FILE__, __LINE__);

    //
    // Compute reference CPU BC solution for source-distance
    //
    if (reference_check_bc_values != NULL) {
        if (ref_filename.empty()) {
            printf("compute ref value\n");
            RefCPUBC(
                    graph,
                    reference_check_bc_values,
                    reference_check_ebc_values,
                    reference_check_sigmas,
                    src);
            printf("\n");
        } else {
            std::ifstream fin;
            fin.open(ref_filename.c_str(), std::ios::binary);
            for ( int i = 0; i < graph.nodes; ++i )
            {
                fin.read(reinterpret_cast<char*>(&reference_check_bc_values[i]), sizeof(Value));
            }
            fin.close();
        }
    }

    double              avg_duty = 0.0;

    // Perform BC
    CpuTimer cpu_timer;
    VertexId start_src;
    VertexId end_src;
    if (src == -1)
    {
        start_src = 0;
        end_src = graph.nodes;
    }
    else
    {
        start_src = src;
        end_src = src+1;
    }


    cpu_timer.Start();
    for (VertexId i = start_src; i < end_src; ++i)
    {
        util::GRError(csr_problem->Reset(i, bc_enactor->GetFrontierType(), max_queue_sizing), "BC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(bc_enactor ->Enact(context, csr_problem, i, max_grid_size), "BC Problem Enact Failed", __FILE__, __LINE__);
    }

    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        util::SetDevice(gpu_idx[gpu]);
        util::MemsetScaleKernel<<<128, 128>>>
            (csr_problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE), (Value)0.5f, (int)(csr_problem->sub_graphs[gpu].nodes));
    }
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();

    bc_enactor->GetStatistics(avg_duty);

    // Copy out results
    util::GRError(csr_problem->Extract(h_sigmas, h_bc_values, h_ebc_values), "BC Problem Data Extraction Failed", __FILE__, __LINE__);
    /*printf("edge bc values: %d\n", graph.edges);
    for (int i = 0; i < graph.edges; ++i) {
        printf("%5f, %5f\n", h_ebc_values[i], reference_check_ebc_values[i]);
    }
    printf("edge bc values end\n");*/

    // Verify the result
    if (reference_check_bc_values != NULL) {
        printf("Validity BC Value: ");
        int num_error = CompareResults(h_bc_values, reference_check_bc_values, graph.nodes,
                       true);
        if (num_error > 0)
            printf("Number of errors occurred: %d\n", num_error);
        printf("\n");
    }
    if (reference_check_ebc_values != NULL) {
        printf("Validity Edge BC Value: ");
        CompareResults(h_ebc_values, reference_check_ebc_values, graph.edges,
                       true);
        printf("\n");
    }
    if (reference_check_sigmas != NULL) {
        printf("Validity Sigma: ");
        CompareResults(h_sigmas, reference_check_sigmas, graph.nodes, true);
        printf("\n");
    }

    // Display Solution
    DisplaySolution(h_sigmas, h_bc_values, graph.nodes);

    printf("GPU BC finished in %lf msec.\n", elapsed);
    if (avg_duty != 0)
        printf("\n avg CTA duty: %.2f%% \n", avg_duty * 100);
    
    // Cleanup
    if (csr_problem         ) {delete   csr_problem         ; csr_problem          = NULL;}
    if (bc_enactor          ) {delete   bc_enactor          ; bc_enactor           = NULL;}
    if (reference_sigmas    ) {delete[] reference_sigmas    ; reference_sigmas     = NULL;}
    if (reference_bc_values ) {delete[] reference_bc_values ; reference_bc_values  = NULL;}
    if (reference_ebc_values) {delete[] reference_ebc_values; reference_ebc_values = NULL;}
    if (h_sigmas            ) {delete[] h_sigmas            ; h_sigmas             = NULL;}
    if (h_bc_values         ) {delete[] h_bc_values         ; h_bc_values          = NULL;}
    if (h_ebc_values        ) {delete[] h_ebc_values        ; h_ebc_values         = NULL;}

    //cudaDeviceSynchronize();
}

template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    CommandLineArgs             &args,
    int                         num_gpus,
    ContextPtr                  *context,
    int                         *gpu_idx)
{
    VertexId    src              = -1;    // Use whatever the specified graph-type's default is
    std::string src_str;
    std::string ref_filename;
    bool        instrumented     = false; // Whether or not to collect instrumentation from kernels
    int         max_grid_size    = 0;     // maximum grid size (0: leave it up to the enactor)
    //int         num_gpus         = 1;     // Number of GPUs for multi-gpu enactor to use
    double      max_queue_sizing = 1.0;   // Maximum size scaling factor for work queues (e.g., 1.0 creates n and m-element vertex and edge frontiers).
    std::string partition_method = "random";

    instrumented = args.CheckCmdLineFlag("instrumented");
    args.GetCmdLineArgument("src", src_str);
    args.GetCmdLineArgument("ref-file", ref_filename);
    if (src_str.empty()) {
        src = -1;
    } else {
        args.GetCmdLineArgument("src", src);
    }

    g_quick = args.CheckCmdLineFlag("quick");
    args.GetCmdLineArgument("queue-sizing", max_queue_sizing);
    g_verbose = args.CheckCmdLineFlag("v");
    if (args.CheckCmdLineFlag  ("partition_method"))
        args.GetCmdLineArgument("partition_method",partition_method);

    if (instrumented) {
        RunTests<VertexId, Value, SizeT, true>(
            graph,
            src,
            ref_filename,
            max_grid_size,
            num_gpus,
            max_queue_sizing,
            context,
            partition_method,
            gpu_idx);
    } else {
        RunTests<VertexId, Value, SizeT, false>(
            graph,
            src,
            ref_filename,
            max_grid_size,
            num_gpus,
            max_queue_sizing,
            context,
            partition_method,
            gpu_idx);
    }

}



/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int        num_gpus = 0;
    int        *gpu_idx = NULL;
    ContextPtr *context = NULL;

    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

    //DeviceInit(args);
    //cudaSetDeviceFlags(cudaDeviceMapHost);
    if (args.CheckCmdLineFlag  ("device"))
    {
        std::vector<int> gpus;
        args.GetCmdLineArguments<int>("device",gpus);
        num_gpus   = gpus.size();
        gpu_idx    = new int[num_gpus];
        for (int i=0;i<num_gpus;i++)
            gpu_idx[i] = gpus[i];
    } else {
        num_gpus   = 1;
        gpu_idx    = new int[num_gpus];
        gpu_idx[0] = 0;
    }
    context  = new ContextPtr[num_gpus];
    printf("Using %d gpus: ", num_gpus);
    for (int i=0;i<num_gpus;i++)
    {
        printf(" %d ", gpu_idx[i]);
        context[i] = mgpu::CreateCudaDevice(gpu_idx[i]);
    }
    printf("\n"); fflush(stdout);

    //srand(0);                                                                     // Presently deterministic
    //srand(time(NULL));

    // Parse graph-contruction params
    g_undirected = true;

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

        typedef int VertexId;  // Use as the node identifier type
        typedef float Value;   // Use as the value type
        typedef int SizeT;     // Use as the graph size type
        Csr<VertexId, Value, SizeT> csr(false); // default value for stream_from_host is false

        if (graph_args < 1) { Usage(); return 1; }
        char *market_filename = (graph_args == 2 || graph_args == 3) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
                market_filename,
                csr,
                g_undirected,
                false) != 0)    //no inverse graph
        {
            return 1;
        }

        csr.PrintHistogram();
        //csr.DisplayGraph();
        fflush(stdout);

        // Run tests
        RunTests(csr, args, num_gpus, context, gpu_idx);

    } else {

        // Unknown graph type
        fprintf(stderr, "Unspecified graph type\n");
        return 1;

    }

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
