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
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::pr;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

template <typename VertexId, typename Value>
struct RankPair {
    VertexId        vertex_id;
    Value           page_rank;

    RankPair(VertexId vertex_id, Value page_rank) : vertex_id(vertex_id), page_rank(page_rank) {}
};

template<typename RankPair>
bool PRCompare(
    RankPair elem1,
    RankPair elem2)
{
    return elem1.page_rank > elem2.page_rank;
}

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
    float               delta    ;//           = 0.85f;  // Use whatever the specified graph-type's default is
    float               error    ;//           = 0.01f;  // Error threshold
    int                 max_iter ;//           = 20;
    int                 traversal_mode;

    Test_Parameter() {
        delta = 0.85f;
        error = 0.01f;
        max_iter = 20;
        src      = -1;
        traversal_mode = -1;
    }   
    ~Test_Parameter(){   }   

    void Init(CommandLineArgs &args)
    {   
        TestParameter_Base::Init(args);
        args.GetCmdLineArgument("delta", delta);
        args.GetCmdLineArgument("error", error);
        args.GetCmdLineArgument("max-iter", max_iter);
        args.GetCmdLineArgument("src", src);
        args.GetCmdLineArgument("traversal-mode", traversal_mode);
   }   
};

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
 void Usage()
 {
 printf("\ntest_pr <graph type> <graph type args> [--device=<device_index>] "
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

 /**
  * @brief Displays the BFS result (i.e., distance from source)
  *
  * @param[in] source_path Search depth from the source for each node.
  * @param[in] nodes Number of nodes in the graph.
  */
 template<typename VertexId, typename Value, typename SizeT>
 void DisplaySolution(VertexId *node_id, Value *rank, SizeT nodes)
 { 
     // Print out at most top 10 largest components
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
 *
 * \return Zero if two vectors are exactly the same, non-zero if there is any difference.
 *
 */
template <typename SizeT>
int CompareResults_(float* computed, float* reference, SizeT len, bool verbose = true)
{
    float THRESHOLD = 0.05f;
    int flag = 0;
    for (SizeT i = 0; i < len; i++) {

        // Use relative error rate here.
        bool is_right = true;
        if (fabs(computed[i]) < 0.01f && fabs(reference[i]-1) < 0.01f) continue;
        if (fabs(computed[i] - 0.0) < 0.01f) {
            if (fabs(computed[i] - reference[i]) > THRESHOLD)
                is_right = false;
        } else {
            if (fabs((computed[i] - reference[i])/reference[i]) > THRESHOLD)
                is_right = false;
        }   
        if (!is_right && flag == 0) {
            printf("\nINCORRECT: [%lu]: ", (unsigned long) i); 
            PrintValue<float>(computed[i]);
            printf(" != ");
            PrintValue<float>(reference[i]);

            if (verbose) {
                printf("\nresult[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    PrintValue<float>(computed[j]);
                    printf(", ");
                }   
                printf("...]");
                printf("\nreference[...");
                for (size_t j = (i >= 5) ? i - 5 : 0; (j < i + 5) && (j < len); j++) {
                    PrintValue<float>(reference[j]);
                    printf(", ");
                }   
                printf("...]");
            }   
            flag += 1;
            //return flag;
        }   
        if (!is_right && flag > 0) flag += 1;
    }   
    printf("\n");
    if (!flag)
        printf("CORRECT");
    return flag;
}
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
    Csr<VertexId, Value, SizeT> &graph,
    double              elapsed,
    long long           total_queued,
    double              avg_duty)
{
    
    // Display test name
    printf("[%s] finished. ", stats.name);

    // Display the specific sample statistics
    printf(" elapsed: %.3f ms", elapsed);
    printf(", #edges visited: %lld", total_queued);
    if (avg_duty != 0) {
        printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
    }
    printf("\n");
}


/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

 /**
  * @brief A simple CPU-based reference Page Rank implementation.
  *
  * @tparam VertexId
  * @tparam Value
  * @tparam SizeT
  *
  * @param[in] graph Reference to the CSR graph we process on
  * @param[in] rank Host-side vector to store CPU computed labels for each node
  * @param[in] delta delta for computing PR
  * @param[in] error error threshold
  * @param[in] max_iter max iteration to go
  */
 template<
    typename VertexId,
    typename Value,
    typename SizeT>
void SimpleReferencePr(
    Csr<VertexId, Value, SizeT>             &graph,
    VertexId                                *node_id,
    Value                                   *rank,
    Value                                   delta,
    Value                                   error,
    SizeT                                   max_iter,
    bool                                    directed) 
{
    using namespace boost;

    //Preparation
    typedef adjacency_list<vecS, vecS, bidirectionalS, 
                           no_property, property<edge_index_t, int> > Graph;

    Graph g;

    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
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

    if (!directed)
    {
        remove_dangling_links(g);
        printf("finished remove dangling links.\n");
    }

    std::vector<Value> ranks(num_vertices(g));
    page_rank(g,
              make_iterator_property_map(ranks.begin(),
              get(boost::vertex_index, g)),
              boost::graph::n_iterations(max_iter));
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    for (std::size_t i = 0; i < num_vertices(g); ++i) {
        rank[i] = ranks[i];
    }

    //sort the top page ranks
    RankPair<SizeT, Value> *pr_list = 
        (RankPair<SizeT, Value>*)malloc(
            sizeof(RankPair<SizeT, Value>) * num_vertices(g));
     for (int i = 0; i < num_vertices(g); ++i)
     {
         pr_list[i].vertex_id = i;
         pr_list[i].page_rank = rank[i];
     }
     std::stable_sort(pr_list, pr_list + num_vertices(g), PRCompare<RankPair<SizeT, Value> >);

     for (int i = 0; i < num_vertices(g); ++i)
     {
         node_id[i] = pr_list[i].vertex_id;
         rank[i] = pr_list[i].page_rank;
     }

    free(pr_list);
    
    printf("CPU PR finished in %lf msec.\n", elapsed);
}

/**
 * @brief Run PR tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
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
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK>
void RunTests(Test_Parameter *parameter)
{
    
    typedef PRProblem<
        VertexId,
        SizeT,
        Value> PrProblem;

    typedef PREnactor<
        PrProblem,
        INSTRUMENT,
        DEBUG,
        SIZE_CHECK> PrEnactor;

    Csr<VertexId, Value, SizeT>
                 *graph              = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    int           max_grid_size      = parameter -> max_grid_size;
    int           num_gpus           = parameter -> num_gpus;
    double        max_queue_sizing   = parameter -> max_queue_sizing;
    double        max_in_sizing      = parameter -> max_in_sizing;
    ContextPtr   *context            = (ContextPtr*)parameter -> context;
    std::string   partition_method   = parameter -> partition_method;
    int          *gpu_idx            = parameter -> gpu_idx;
    cudaStream_t *streams            = parameter -> streams;
    float         partition_factor   = parameter -> partition_factor;
    int           partition_seed     = parameter -> partition_seed;
    bool          g_quick            = parameter -> g_quick;
    bool          g_stream_from_host = parameter -> g_stream_from_host;
    bool          g_undirected       = parameter -> g_undirected;
    VertexId      src                = parameter -> src;
    Value         delta              = parameter -> delta;
    Value         error              = parameter -> error;
    SizeT         max_iter           = parameter -> max_iter;
    int           traversal_mode     = parameter -> traversal_mode;
    size_t       *org_size           = new size_t  [num_gpus];
    // Allocate host-side label array (for both reference and gpu-computed results)
    Value        *reference_rank     = new Value   [graph->nodes];
    Value        *h_rank             = new Value   [graph->nodes];
    VertexId     *h_node_id          = new VertexId[graph->nodes];
    VertexId     *reference_node_id  = new VertexId[graph->nodes];
    Value        *reference_check    = (g_quick) ? NULL : reference_rank;

    for (int gpu=0; gpu<num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    // Allocate BFS enactor map
    PrEnactor* enactor = new PrEnactor(num_gpus, gpu_idx);

    // Allocate problem on GPU
    PrProblem *problem = new PrProblem;
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
        partition_seed), "Problem pr Initialization Failed", __FILE__, __LINE__);
    util::GRError(enactor->Init(context, problem, traversal_mode, /*max_iter,*/ max_grid_size), "PR Enactor Init failed", __FILE__, __LINE__);

    Stats *stats = new Stats("GPU PageRank");

    long long           total_queued = 0;
    double              avg_duty = 0.0;

    // Perform BFS
    GpuTimer gpu_timer;

    util::GRError(problem->Reset(src, delta, error, max_iter, enactor->GetFrontierType(), max_queue_sizing), "pr Problem Data Reset Failed", __FILE__, __LINE__);
    util::GRError(enactor->Reset(), "PR Enactor Reset Reset failed", __FILE__, __LINE__);
    
    printf("_________________________________________\n");fflush(stdout);
    gpu_timer.Start();
    util::GRError(enactor->Enact(traversal_mode), "pr Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();
    printf("-----------------------------------------\n");fflush(stdout);

    enactor->GetStatistics(total_queued, avg_duty);

    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(problem->Extract(h_rank, h_node_id), "PageRank Problem Data Extraction Failed", __FILE__, __LINE__);

    float total_pr = 0;
    for (int i = 0; i < graph->nodes; ++i)
    {
        total_pr += h_rank[i];
    }
    printf("Total rank : %f\n", total_pr);

    //
    // Compute reference CPU PR solution for source-distance
    //
    if (reference_check != NULL && total_pr > 0)
    {
        printf("compute ref value\n");
        SimpleReferencePr <VertexId, Value, SizeT> (
                *graph,
                reference_node_id,
                reference_check,
                delta,
                error,
                max_iter,
                !g_undirected);
        printf("\n");
    }

    // Verify the result
    if (reference_check != NULL && total_pr > 0) {
        printf("Validity Rank: ");
        int errors_count = CompareResults_(h_rank, reference_check, graph->nodes, true);
        if (errors_count > 0) printf("number of errors : %lld\n",(long long) errors_count);

        /*printf("Validity node_id: ");
        errors_count = CompareResults(h_node_id, reference_node_id, graph->nodes, true);
        if (errors_count > 0) printf("number of errors : %lld\n", (long long) errors_count);*/
    }
    printf("\nFirst 40 labels of the GPU result."); 
    // Display Solution
    DisplaySolution(h_node_id, h_rank, graph->nodes);

    DisplayStats(
        *stats,
        h_rank,
        *graph,
        elapsed,
        total_queued,
        avg_duty);

    printf("\n\tMemory Usage(B)\t");
    for (int gpu=0;gpu<num_gpus;gpu++)
    if (num_gpus>1) {if (gpu!=0) printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1",gpu,gpu,gpu,gpu); else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);}
    else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
    if (num_gpus>1) printf(" #keys%d",num_gpus);
    printf("\n");
    double max_queue_sizing_[2] = {0,0}, max_in_sizing_=0;
    for (int gpu=0;gpu<num_gpus;gpu++)
    {   
        size_t gpu_free,dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&gpu_free,&dummy);
        printf("GPU_%d\t %ld",gpu_idx[gpu],org_size[gpu]-gpu_free);
        for (int i=0;i<num_gpus;i++)
        {   
            for (int j=0; j<2; j++)
            {   
                SizeT x=problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
                printf("\t %lld", (long long) x); 
                double factor = 1.0*x/(num_gpus>1?problem->graph_slices[gpu]->in_counter[i]:problem->graph_slices[gpu]->nodes);
                if (factor > max_queue_sizing_[j]) max_queue_sizing_[j]=factor;
            }   
            if (num_gpus>1 && i!=0 )
            for (int t=0;t<2;t++)
            {   
                SizeT x=problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                printf("\t %lld", (long long) x); 
                double factor = 1.0*x/problem->graph_slices[gpu]->in_counter[i];
                if (factor > max_in_sizing_) max_in_sizing_=factor;
            }   
        }   
        if (num_gpus>1) printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
        printf("\n");
    }   
    printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
    if (num_gpus>1) printf("\t in_sizing =\t %lf", max_in_sizing_);
    printf("\n");

    // Cleanup
    if (stats            ) {delete   stats            ; stats             = NULL;}
    if (org_size         ) {delete   org_size         ; org_size          = NULL;}
    if (problem          ) {delete   problem          ; problem           = NULL;}
    if (enactor          ) {delete   enactor          ; enactor           = NULL;}
    if (reference_rank   ) {delete[] reference_rank   ; reference_rank    = NULL;}
    if (reference_node_id) {delete[] reference_node_id; reference_node_id = NULL;}
    if (h_rank           ) {delete[] h_rank           ; h_rank            = NULL;}
    if (h_node_id        ) {delete[] h_node_id        ; h_node_id         = NULL;}

    //cudaDeviceSynchronize();
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG>
void RunTests_size_check(Test_Parameter *parameter)
{
    if (parameter->size_check) RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        true > (parameter);
   else RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        false> (parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT>
void RunTests_debug(Test_Parameter *parameter)
{
    if (parameter->debug) RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        true > (parameter);
    else RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        false> (parameter);
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
void RunTests(Test_Parameter* parameter)
{
    if (parameter->instrumented) RunTests_debug
        <VertexId, Value, SizeT,
        true > (parameter);
    else RunTests_debug
        <VertexId, Value, SizeT,
        false> (parameter);
}



/******************************************************************************
* Main
******************************************************************************/

int cpp_main( int argc, char** argv)
{
    CommandLineArgs  args(argc, argv);
    int              num_gpus = 0;
    int             *gpu_idx  = NULL;
    ContextPtr      *context  = NULL;
    cudaStream_t    *streams  = NULL;
    //bool             g_undirected = false; //Does not make undirected graph

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
    streams  = new cudaStream_t[num_gpus * num_gpus * 2]; 
    context  = new ContextPtr  [num_gpus * num_gpus];
    printf("Using %d gpus: ", num_gpus);
    for (int gpu=0;gpu<num_gpus;gpu++)
    {   
        printf(" %d ", gpu_idx[gpu]);
        util::SetDevice(gpu_idx[gpu]);
        for (int i=0;i<num_gpus*2;i++)
        {   
            int _i = gpu*num_gpus*2+i;
            util::GRError(cudaStreamCreate(&streams[_i]), "cudaStreamCreate failed.", __FILE__, __LINE__);
            if (i<num_gpus) context[gpu*num_gpus+i] = mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu], streams[_i]);
        }   
    }   
    printf("\n"); fflush(stdout);

    // Parse graph-contruction params
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
    Test_Parameter *parameter = new Test_Parameter;
    parameter -> Init(args);
    parameter -> num_gpus    = num_gpus;
    parameter -> context     = context;
    parameter -> gpu_idx     = gpu_idx;
    parameter -> streams     = streams;

    typedef int VertexId;							// Use as the node identifier type
    typedef float Value;								// Use as the value type
    typedef int SizeT;								// Use as the graph size type
    Csr<VertexId, Value, SizeT> graph(false);         // default value for stream_from_host is false

	if (graph_type == "market") {

		// Matrix-market coordinate-formatted graph file

        if (graph_args < 1) { Usage(); return 1; }
        char *market_filename = (graph_args == 2) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
			market_filename, 
			graph, 
			parameter->g_undirected,
			false) != 0) // no inverse graph
		{
			return 1;
		}

	} else if (graph_type == "rmat")
    {
        // parse rmat parameters
        SizeT rmat_nodes = 1 << 10;
        SizeT rmat_edges = 1 << 10;
        SizeT rmat_scale = 10;
        SizeT rmat_edgefactor = 48;
        double rmat_a = 0.57;
        double rmat_b = 0.19;
        double rmat_c = 0.19;
        double rmat_d = 1-(rmat_a+rmat_b+rmat_c);
        int    rmat_seed = -1;

        args.GetCmdLineArgument("rmat_scale", rmat_scale);
        rmat_nodes = 1 << rmat_scale;
        args.GetCmdLineArgument("rmat_nodes", rmat_nodes);
        args.GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
        rmat_edges = rmat_nodes * rmat_edgefactor;
        args.GetCmdLineArgument("rmat_edges", rmat_edges);
        args.GetCmdLineArgument("rmat_a", rmat_a);
        args.GetCmdLineArgument("rmat_b", rmat_b);
        args.GetCmdLineArgument("rmat_c", rmat_c);
        rmat_d = 1-(rmat_a+rmat_b+rmat_c);
        args.GetCmdLineArgument("rmat_d", rmat_d);
        args.GetCmdLineArgument("rmat_seed", rmat_seed);

        CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::BuildRmatGraph<false>(
                rmat_nodes,
                rmat_edges,
                graph,
                parameter->g_undirected,
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                1,
                1,
                rmat_seed) != 0)
        {
            return 1;
        }
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        printf("graph generated: %.3f ms, a = %.3f, b = %.3f, c = %.3f, d = %.3f\n", elapsed, rmat_a, rmat_b, rmat_c, rmat_d);
    } else if (graph_type == "rgg") {

        SizeT rgg_nodes = 1 << 10;
        SizeT rgg_scale = 10;
        double rgg_thfactor  = 0.55;
        double rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        double rgg_vmultipiler = 1;
        int    rgg_seed        = -1;

        args.GetCmdLineArgument("rgg_scale", rgg_scale);
        rgg_nodes = 1 << rgg_scale;
        args.GetCmdLineArgument("rgg_nodes", rgg_nodes);
        args.GetCmdLineArgument("rgg_thfactor", rgg_thfactor);
        rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        args.GetCmdLineArgument("rgg_threshold", rgg_threshold);
        args.GetCmdLineArgument("rgg_vmultipiler", rgg_vmultipiler);
        args.GetCmdLineArgument("rgg_seed", rgg_seed);

        CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::BuildRggGraph<false>(
            rgg_nodes,
            graph,
            rgg_threshold,
            parameter->g_undirected,
            rgg_vmultipiler,
            1,
            rgg_seed) !=0)
        {
            return 1;
        }
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        printf("graph generated: %.3f ms, threshold = %.3lf, vmultipiler = %.3lf\n", elapsed, rgg_threshold, rgg_vmultipiler);
    } else {

		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;

	}

    parameter -> graph       = &graph;
    if (parameter -> traversal_mode == -1)
        parameter -> traversal_mode = graph.GetAverageDegree()>3 ? 0 : 1;

    graph.PrintHistogram();

    // Run tests
    RunTests<VertexId, Value, SizeT>(parameter);

	return 0;
}
