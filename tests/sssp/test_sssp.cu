// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief Simple test driver program for single source shorest path.
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
 printf("\ntest_sssp <graph type> <graph type args> [--device=<device_index>] "
        "[--undirected] [--instrumented] [--src=<source index>] [--quick]\n"
        "[--v] [mark-pred] [--queue-sizing=<scale factor>] "
        "[--in-sizing=<in/out queue scale factor>] [--disable-size-check] "
        "[--grid-size=<grid size>] [partition_method=random / biasrandom / clustered / metis]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
        "  --undirected If set then treat the graph as undirected.\n"
        "  --instrumented If set then kernels keep track of queue-search_depth\n"
        "  and barrier duty (a relative indicator of load imbalance.)\n"
        "  --src Begins SSSP from the vertex <source index>. If set as randomize\n"
        "  then will begin with a random source vertex.\n"
        "  If set as largestdegree then will begin with the node which has\n"
        "  largest degree.\n"
        "  --quick If set will skip the CPU validation code.\n"
        "  --v Whether to show debug info.\n"
        "  --mark-pred If set then keep not only label info but also predecessor info.\n"
        "  --queue-sizing Allocates a frontier queue sized at (graph-edges * <scale factor>).\n"
        );
 }

 /**
  * @brief Displays the SSSP result (i.e., distance from source)
  *
  * @param[in] source_path Search depth from the source for each node.
  * @param[in] preds Predecessor node id for each node.
  * @param[in] nodes Number of nodes in the graph.
  * @param[in] MARK_PREDECESSORS Whether to show predecessor of each node.
  */
 template<typename VertexId, typename SizeT>
 void DisplaySolution(VertexId *source_path, SizeT nodes)
 {
    if (nodes > 40)
        nodes = 40;
    printf("[");
    for (VertexId i = 0; i < nodes; ++i) {
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
    Stats(const char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
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
    for (VertexId i = 0; i < graph.nodes; ++i) {
        if (h_labels[i] < util::MaxValue<VertexId>()) {
            ++nodes_visited;
            edges_visited += graph.row_offsets[i+1] - graph.row_offsets[i];
        }
    }

    double redundant_work = 0.0;
    if (total_queued > 0) {
        redundant_work = ((double) total_queued - edges_visited) / edges_visited;        // measure duplicate edges put through queue
    }
    redundant_work *= 100;

    // Display test name
    printf("[%s] finished. ", stats.name);

    // Display statistics
    if (nodes_visited < 5) {
        printf("Fewer than 5 vertices visited.\n");
    } else {
        // Display the specific sample statistics
        double m_teps = (double) edges_visited / (elapsed * 1000.0);
        printf(" elapsed: %.3f ms, rate: %.3f MiEdges/s", elapsed, m_teps);
        if (search_depth != 0) printf(", search_depth: %lld", (long long) search_depth);
        if (avg_duty != 0) {
            printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
        }
        printf("\n src: %lld, nodes_visited: %lld, edges visited: %lld",
            (long long) src, (long long) nodes_visited, (long long) edges_visited);
        if (total_queued > 0) {
            printf(", total queued: %lld", total_queued);
        }
        if (redundant_work > 0) {
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
    const Csr<VertexId, Value, SizeT>       &graph,
    Value                                   *node_values,
    VertexId                                *node_preds,
    VertexId                                src)
{
    using namespace boost; 
    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS,
            no_property, property <edge_weight_t, unsigned int> > Graph;
    typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typedef graph_traits<Graph>::edge_descriptor edge_descriptor;

    typedef std::pair<unsigned int, unsigned int> Edge;

    Edge* edges = (Edge*)malloc(sizeof(Edge)*graph.edges);
    unsigned int *weight = (unsigned int*)malloc(sizeof(unsigned int)*graph.edges);

    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
        {
            edges[j] = Edge(i, graph.column_indices[j]);
            weight[j] = graph.edge_values[j];
        }
    }

    Graph g(edges, edges + graph.edges, weight, graph.nodes);

    std::vector<unsigned int> d(graph.nodes);
    std::vector<vertex_descriptor> p(graph.nodes);
    vertex_descriptor s = vertex(src, g);

    property_map<Graph, vertex_index_t>::type indexmap = get(vertex_index, g);

    //
    // Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    if (MARK_PREDECESSORS)
        dijkstra_shortest_paths(g,
                            s,
                            predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
                            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
    else
        dijkstra_shortest_paths(g,
                            s,
                            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    printf("CPU SSSP finished in %lf msec.\n", elapsed);

    Coo<unsigned int, unsigned int>* sort_dist = NULL;
    Coo<unsigned int, unsigned int>* sort_pred = NULL;
    sort_dist = (Coo<unsigned int, unsigned int>*)malloc(sizeof(Coo<unsigned int, unsigned int>) * graph.nodes);
    if (MARK_PREDECESSORS)
        sort_pred = (Coo<unsigned int, unsigned int>*)malloc(sizeof(Coo<unsigned int, unsigned int>) * graph.nodes);

    graph_traits < Graph >::vertex_iterator vi, vend;
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    {
        sort_dist[(*vi)].row = (*vi);
        sort_dist[(*vi)].col = d[(*vi)];  
    }
    std::stable_sort(sort_dist, sort_dist + graph.nodes, RowFirstTupleCompare<Coo<unsigned int, unsigned int> >);

    if (MARK_PREDECESSORS) {
        for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
        {
            sort_pred[(*vi)].row = (*vi);
            sort_pred[(*vi)].col = p[(*vi)];
        }
        std::stable_sort(sort_pred, sort_pred + graph.nodes, RowFirstTupleCompare<Coo<unsigned int, unsigned int> >);
    }

    for (int i = 0; i < graph.nodes; ++i) {
        node_values[i] = sort_dist[i].col;
    }
    if (MARK_PREDECESSORS)
        for (int i = 0; i < graph.nodes; ++i) {
            node_preds[i] = sort_pred[i].col;
        }

    free(sort_dist);
    if (MARK_PREDECESSORS) free(sort_pred);
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
 * @param[in] num_gpus Number of GPUs
 * @param[in] max_queue_sizing Scaling factor used in edge mapping
 *
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool MARK_PREDECESSORS>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    VertexId     src,
    int          max_grid_size,
    float        max_queue_sizing,
    float        max_in_sizing,
    int          num_gpus,
    int          delta_factor,
    int          iterations,
    ContextPtr   *context,
    std::string  partition_method,
    int          *gpu_idx,
    cudaStream_t *streams,
    bool         size_check = true)
{
        size_t *org_size = new size_t[num_gpus];

        typedef SSSPProblem<
            VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS> Problem;

        // Allocate host-side label array (for both reference and gpu-computed results)
        Value     *reference_labels       = new Value[graph.nodes];
        Value     *h_labels               = new Value[graph.nodes];
        Value     *reference_check_label  = (g_quick) ? NULL : reference_labels;
        VertexId  *reference_preds        = NULL;
        VertexId  *h_preds                = NULL;
        VertexId  *reference_check_pred   = NULL;

        if (MARK_PREDECESSORS) {
            reference_preds       = new VertexId[graph.nodes];
            h_preds               = new VertexId[graph.nodes];
            reference_check_pred  = (g_quick) ? NULL : reference_preds;
        }
    
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            size_t dummy;
            cudaSetDevice(gpu_idx[gpu]);
            cudaMemGetInfo(&(org_size[gpu]),&dummy);
        }
            
        // Allocate SSSP enactor map
        SSSPEnactor<Problem, INSTRUMENT>* sssp_enactor
            = new SSSPEnactor<Problem, INSTRUMENT>(g_verbose, num_gpus, gpu_idx);

        // Allocate problem on GPU
        Problem *csr_problem = new Problem;
        util::GRError(csr_problem->Init(
            g_stream_from_host,
            graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method,
            streams,
            delta_factor,
            max_queue_sizing,
            max_in_sizing), "Problem SSSP Initialization Failed", __FILE__, __LINE__);
        util::GRError(sssp_enactor->Init (context, csr_problem, max_grid_size, size_check), "SSSP Enactor init failed", __FILE__, __LINE__);
        //
        // Compute reference CPU SSSP solution for source-distance
        //
        if (reference_check_label != NULL)
        {
            printf("compute ref value\n");
            SimpleReferenceSssp<VertexId, Value, SizeT, MARK_PREDECESSORS>(
                    graph,
                    reference_check_label,
                    reference_check_pred,
                    src);
            printf("\n");
        }

        Stats      *stats       = new Stats("GPU SSSP");
        long long  total_queued = 0;
        VertexId   search_depth = 0;
        double     avg_duty     = 0.0;
        float      elapsed      = 0.0f;

        // Perform SSSP
        CpuTimer cpu_timer;

        for (int iter = 0; iter < iterations; ++iter)
        {
            util::GRError(csr_problem->Reset(src, sssp_enactor->GetFrontierType(), max_queue_sizing), "SSSP Problem Data Reset Failed", __FILE__, __LINE__); 
            util::GRError(sssp_enactor->Reset(), "SSSP Enactor Reset failed", __FILE__, __LINE__);

            cpu_timer.Start();
            util::GRError(sssp_enactor->Enact(src), "SSSP Problem Enact Failed", __FILE__, __LINE__);
            cpu_timer.Stop();
            elapsed += cpu_timer.ElapsedMillis();
        }
        elapsed /= iterations;

        sssp_enactor->GetStatistics(total_queued, search_depth, avg_duty);

        // Copy out results
        util::GRError(csr_problem->Extract(h_labels, h_preds), "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

        for (SizeT i=0; i<graph.nodes;i++)
        if (reference_check_label[i]==-1) reference_check_label[i]=util::MaxValue<Value>();

        // Verify the result
        if (reference_check_label != NULL) {
            printf("Label Validity: ");
            int error_num = CompareResults(h_labels, reference_check_label, graph.nodes, true);
            if (error_num > 0)
                printf("%d errors occurred.\n", error_num);
        }
        
        // Display Solution
        printf("\nFirst 40 labels of the GPU result.\n"); 
        DisplaySolution(h_labels, graph.nodes);
        printf("\nFirst 40 labels of the reference CPU result.\n"); 
        DisplaySolution(reference_check_label, graph.nodes);

        if (MARK_PREDECESSORS) {
            printf("\nFirst 40 preds of the GPU result.\n"); 
            DisplaySolution(h_preds, graph.nodes);
            printf("\nFirst 40 preds of the reference CPU result (could be different because the paths are not unique).\n"); 
            DisplaySolution(reference_check_pred, graph.nodes);
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

        printf("\n\tMemory Usage(B)\t");
        for (int gpu=0;gpu<num_gpus;gpu++)
        if (num_gpus>1) {if (gpu!=0) printf(" #keys%d\t #ins%d,0\t #ins%d,1",gpu,gpu,gpu); else printf(" $keys%d", gpu);}
        else printf(" #keys%d", gpu);
        if (num_gpus>1) printf(" #keys%d",num_gpus);
        printf("\n");
        double max_key_sizing=0, max_in_sizing_=0;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            size_t gpu_free,dummy;
            cudaSetDevice(gpu_idx[gpu]);
            cudaMemGetInfo(&gpu_free,&dummy);
            printf("GPU_%d\t %ld",gpu_idx[gpu],org_size[gpu]-gpu_free);
            for (int i=0;i<num_gpus;i++)
            {
                SizeT x=csr_problem->graph_slices[gpu]->frontier_queues[i].keys[0].GetSize();
                printf("\t %d", x);
                double factor = 1.0*x/(num_gpus>1?csr_problem->graph_slices[gpu]->in_counter[i]:csr_problem->graph_slices[gpu]->nodes);
                if (factor > max_key_sizing) max_key_sizing=factor;
                if (num_gpus>1 && i!=0 )
                for (int t=0;t<2;t++)
                {
                    x=csr_problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                    printf("\t %d", x);
                    factor = 1.0*x/csr_problem->graph_slices[gpu]->in_counter[i];
                    if (factor > max_in_sizing_) max_in_sizing_=factor;
                }
            }
            if (num_gpus>1) printf("\t %d",csr_problem->graph_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize());
            printf("\n");
        }
        printf("\t key_sizing =\t %lf", max_key_sizing);
        if (num_gpus>1) printf("\t in_sizing =\t %lf", max_in_sizing_);
        printf("\n");

        // Cleanup
        if (org_size        ) {delete[] org_size        ; org_size         = NULL;}
        if (stats           ) {delete   stats           ; stats            = NULL;}
        if (sssp_enactor    ) {delete   sssp_enactor    ; sssp_enactor     = NULL;}
        if (csr_problem     ) {delete   csr_problem     ; csr_problem      = NULL;}
        if (reference_labels) {delete[] reference_labels; reference_labels = NULL;}
        if (h_labels        ) {delete[] h_labels        ; h_labels         = NULL;}
        if (reference_preds ) {delete[] reference_preds ; reference_preds  = NULL;}
        if (h_preds         ) {delete[] h_preds         ; h_preds          = NULL;}

        //cudaDeviceSynchronize();
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
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> &graph,
    CommandLineArgs             &args,
    int                         num_gpus,
    ContextPtr                  *context,
    int                         *gpu_idx,
    cudaStream_t                *streams)
{
    VertexId            src                 = -1;           // Use whatever the specified graph-type's default is
    std::string         src_str;
    bool                instrumented        = false;        // Whether or not to collect instrumentation from kernels
    int                 max_grid_size       = 0;            // maximum grid size (0: leave it up to the enactor)
    //int                 num_gpus            = 1;            // Number of GPUs for multi-gpu enactor to use
    float               max_queue_sizing    = 1.0;
    float               max_in_sizing       = 1.0;
    bool                mark_pred           = false;
    std::string         partition_method    = "random";
    int                 iterations          = 1;
    bool                disable_size_check  = false;
    int                 delta_factor        = 16;

    instrumented = args.CheckCmdLineFlag("instrumented");
    disable_size_check = args.CheckCmdLineFlag("disable-size-check");
    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty()) {
        src = 0;
    } else if (src_str.compare("randomize") == 0) {
        src = graphio::RandomNode(graph.nodes);
    } else if (src_str.compare("largestdegree") == 0) {
        int temp;
        src = graph.GetNodeWithHighestDegree(temp);
    } else {
        args.GetCmdLineArgument("src", src);
    }

    g_verbose = args.CheckCmdLineFlag("v"        );
    g_quick   = args.CheckCmdLineFlag("quick"    );
    mark_pred = args.CheckCmdLineFlag("mark-pred");
    args.GetCmdLineArgument("iteration-num", iterations      );
    args.GetCmdLineArgument("queue-sizing",  max_queue_sizing);
    args.GetCmdLineArgument("in-sizing",     max_in_sizing   );
    args.GetCmdLineArgument("grid-size",     max_grid_size   );
    args.GetCmdLineArgument("delta-factor",  delta_factor    );
    if (args.CheckCmdLineFlag  ("partition_method")) 
        args.GetCmdLineArgument("partition_method",partition_method);

    if (mark_pred) {
        if (instrumented) {
            RunTests<VertexId, Value, SizeT, true, true>(
                    graph,
                    src,
                    max_grid_size,
                    max_queue_sizing,
                    max_in_sizing,
                    num_gpus,
                    delta_factor,
                    iterations,
                    context,
                    partition_method,
                    gpu_idx,
                    streams,
                    !disable_size_check);
        } else {
            RunTests<VertexId, Value, SizeT, false, true>(
                    graph,
                    src,
                    max_grid_size,
                    max_queue_sizing,
                    max_in_sizing,
                    num_gpus,
                    delta_factor,
                    iterations,
                    context,
                    partition_method,
                    gpu_idx,
                    streams,
                    !disable_size_check);
        }
    } else {
        if (instrumented) {
            RunTests<VertexId, Value, SizeT, true, false>(
                    graph,
                    src,
                    max_grid_size,
                    max_queue_sizing,
                    max_in_sizing,
                    num_gpus,
                    delta_factor,
                    iterations,
                    context,
                    partition_method,
                    gpu_idx,
                    streams,
                    !disable_size_check);
        } else {
            RunTests<VertexId, Value, SizeT, false, false>(
                    graph,
                    src,
                    max_grid_size,
                    max_queue_sizing,
                    max_in_sizing,
                    num_gpus,
                    delta_factor,
                    iterations,
                    context,
                    partition_method,
                    gpu_idx,
                    streams,
                    !disable_size_check);
        }
    }

}



/******************************************************************************
* Main
******************************************************************************/

int cpp_main( int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int          num_gpus = 0;
    int          *gpu_idx = NULL;
    ContextPtr   *context = NULL;
    cudaStream_t *streams = NULL;

    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

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
    streams  = new cudaStream_t[num_gpus * num_gpus *2];
    context  = new ContextPtr  [num_gpus * num_gpus];
    printf("Using %d gpus: ", num_gpus);
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        printf(" %d ", gpu_idx[gpu]);
        util::SetDevice(gpu_idx[gpu]);
        for (int i=0;i<num_gpus*2;i++)
        {
            int _i=gpu*num_gpus*2+i;
            util::GRError(cudaStreamCreate(&streams[_i]), "cudaStreamCreate fialed.",__FILE__,__LINE__);
            if (i<num_gpus) context[gpu*num_gpus+i] = mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu],streams[_i]);
        }
    }
    printf("\n"); fflush(stdout);
    
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

        typedef int VertexId;                   // Use as the node identifier type
        typedef int Value;             // Use as the value type
        typedef int SizeT;                      // Use as the graph size type
        Csr<VertexId, Value, SizeT> csr(false); // default value for stream_from_host is false

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
        
        csr.GetAverageEdgeValue();
        csr.GetAverageDegree();
        int max_degree;
        csr.GetNodeWithHighestDegree(max_degree);
        printf("max degree:%d\n", max_degree);
	
        // Run tests
        RunTests(csr, args, num_gpus, context, gpu_idx, streams);

    } else {
        // Unknown graph type
        fprintf(stderr, "Unspecified graph type\n");
        return 1;
    }

    return 0;
}
