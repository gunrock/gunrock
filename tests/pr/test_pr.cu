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

bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

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
 template<typename Value, typename SizeT>
 void DisplaySolution(Value *rank, SizeT nodes)
 { 
     //sort the top page ranks
     RankPair<SizeT, Value> *pr_list = (RankPair<SizeT, Value>*)malloc(sizeof(RankPair<SizeT, Value>) * nodes);
     Value total_pr = 0;
     for (int i = 0; i < nodes; ++i)
     {
         pr_list[i].vertex_id = i;
         pr_list[i].page_rank = rank[i];
         total_pr += rank[i];
     }
     std::stable_sort(pr_list, pr_list + nodes, PRCompare<RankPair<SizeT, Value> >);

     // Print out at most top 10 largest components
     int top = (nodes < 10) ? nodes : 10;
     printf("Top %d Page Ranks:\n", top);
     for (int i = 0; i < top; ++i)
     {
         printf("Vertex ID: %d, Page Rank: %5f\n", pr_list[i].vertex_id, pr_list[i].page_rank);
     }
     printf("total pr: %5f\n", total_pr);

     free(pr_list);
 }

 /**
  * Performance/Evaluation statistics
  */ 

struct Stats {
    char *name;
    Statistic rate;
    Statistic search_depth;
    Statistic redundant_work;
    Statistic duty;

    Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
    Stats(char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

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
    const Csr<VertexId, Value, SizeT> &graph,
    double              elapsed,
    long long           total_queued,
    double              avg_duty)
{
    
    // Display test name
    printf("[%s] finished. ", stats.name);

    // Display the specific sample statistics
    printf(" elapsed: %.3f ms", elapsed);
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
    const Csr<VertexId, Value, SizeT>       &graph,
    Value                                   *rank,
    Value                                   delta,
    Value                                   error,
    SizeT                                   max_iter) 
{
    using namespace boost;

    //Preparation
    typedef adjacency_list<vecS, vecS, bidirectionalS, no_property, property<edge_index_t, int> > Graph;

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

    remove_dangling_links(g);

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

    printf("CPU BFS finished in %lf msec.\n", elapsed);
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
    bool INSTRUMENT>
void RunTests(
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId src,
    Value delta,
    Value error,
    SizeT max_iter,
    int max_grid_size,
    int num_gpus,
    CudaContext& context)
{
    
    typedef PRProblem<
        VertexId,
        SizeT,
        Value> Problem;

        // Allocate host-side label array (for both reference and gpu-computed results)
        Value    *reference_rank       = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value    *h_rank               = (Value*)malloc(sizeof(Value) * graph.nodes);
        Value    *reference_check        = (g_quick) ? NULL : reference_rank;

        // Allocate BFS enactor map
        PREnactor<INSTRUMENT> pr_enactor(g_verbose);

        // Allocate problem on GPU
        Problem *csr_problem = new Problem;
        util::GRError(csr_problem->Init(
            g_stream_from_host,
            graph,
            num_gpus), "Problem pr Initialization Failed", __FILE__, __LINE__); 

        Stats *stats = new Stats("GPU PageRank");

        long long           total_queued = 0;
        double              avg_duty = 0.0;

        // Perform BFS
        GpuTimer gpu_timer;

        util::GRError(csr_problem->Reset(src, delta, error, pr_enactor.GetFrontierType()), "pr Problem Data Reset Failed", __FILE__, __LINE__);
        gpu_timer.Start();
        util::GRError(pr_enactor.template Enact<Problem>(context, csr_problem, max_iter, max_grid_size), "pr Problem Enact Failed", __FILE__, __LINE__);
        gpu_timer.Stop();

        pr_enactor.GetStatistics(total_queued, avg_duty);

        float elapsed = gpu_timer.ElapsedMillis();

        // Copy out results
        util::GRError(csr_problem->Extract(h_rank), "PageRank Problem Data Extraction Failed", __FILE__, __LINE__);

        float total_pr = 0;
        for (int i = 0; i < graph.nodes; ++i)
        {
            total_pr += h_rank[i];
        }

        //
        // Compute reference CPU PR solution for source-distance
        //
        if (reference_check != NULL && total_pr > 0)
        {
            printf("compute ref value\n");
            SimpleReferencePr(
                    graph,
                    reference_check,
                    delta,
                    error,
                    max_iter);
            printf("\n");
        }

        // Verify the result
        if (reference_check != NULL && total_pr > 0) {
            printf("Validity: ");
            CompareResults(h_rank, reference_check, graph.nodes, true);
        }
        printf("\nFirst 40 labels of the GPU result."); 
        // Display Solution
        DisplaySolution(h_rank, graph.nodes);

        DisplayStats(
            *stats,
            h_rank,
            graph,
            elapsed,
            total_queued,
            avg_duty);


        // Cleanup
        delete stats;
        if (csr_problem) delete csr_problem;
        if (reference_check) free(reference_check);
        if (h_rank) free(h_rank);

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
    Value               delta               = 0.85f;           // Use whatever the specified graph-type's default is
    Value               error               = 0.01f;        // Error threshold
    SizeT               max_iter            = 20;
    bool                instrumented        = false;        // Whether or not to collect instrumentation from kernels
    int                 max_grid_size       = 0;            // maximum grid size (0: leave it up to the enactor)
    int                 num_gpus            = 1;            // Number of GPUs for multi-gpu enactor to use
    VertexId            src                 = -1;

    instrumented = args.CheckCmdLineFlag("instrumented");
    args.GetCmdLineArgument("delta", delta);
    args.GetCmdLineArgument("error", error);
    args.GetCmdLineArgument("max-iter", max_iter);
    args.GetCmdLineArgument("src", src);

    g_quick = args.CheckCmdLineFlag("quick");
    g_verbose = args.CheckCmdLineFlag("v");

    if (instrumented) {
        RunTests<VertexId, Value, SizeT, true>(
                        graph,
                        src,
                        delta,
                        error,
                        max_iter,
                        max_grid_size,
                        num_gpus,
                        context);
    } else {
        RunTests<VertexId, Value, SizeT, false>(
                        graph,
                        src,
                        delta,
                        error,
                        max_iter,
                        max_grid_size,
                        num_gpus,
                        context);
    }
}



/******************************************************************************
* Main
******************************************************************************/

int main( int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
		Usage();
		return 1;
	}

	//DeviceInit(args);
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	int dev = 0;
    args.GetCmdLineArgument("device", dev);
    ContextPtr context = mgpu::CreateCudaDevice(dev);

	//srand(0);									// Presently deterministic
	//srand(time(NULL));

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

		typedef int VertexId;							// Use as the node identifier type
		typedef float Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		Csr<VertexId, Value, SizeT> csr(false);         // default value for stream_from_host is false

		if (graph_args < 1) { Usage(); return 1; }
		char *market_filename = (graph_args == 2) ? argv[2] : NULL;
		if (graphio::BuildMarketGraph<false>(
			market_filename, 
			csr, 
			g_undirected,
			false) != 0) // no inverse graph
		{
			return 1;
		}

		csr.PrintHistogram();

		// Run tests
		RunTests(csr, args, *context);


	} else {

		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;

	}

	return 0;
}
