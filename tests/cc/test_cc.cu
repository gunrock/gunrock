// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_cc.cu
 *
 * @brief Simple test driver program for connected component.
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

// CC includes
#include <gunrock/app/cc/cc_enactor.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

// Boost includes for CPU CC reference algorithms
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::cc;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

//bool g_verbose;
//bool g_undirected;
//bool g_quick;
//bool g_stream_from_host;

template <typename VertexId>
struct CcList {
    VertexId        root;
    unsigned int    histogram;

    CcList(VertexId root, unsigned int histogram) : root(root), histogram(histogram) {}
};

template<typename CcList>
bool CCCompare(
    CcList elem1,
    CcList elem2)
{
    return elem1.histogram > elem2.histogram;
}

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
     Test_Parameter(){   }   
    ~Test_Parameter(){   }   

    void Init(CommandLineArgs &args)
    {   
        TestParameter_Base::Init(args);
    }   
};

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
 void Usage()
 {
 printf("\ntest_cc <graph type> <graph type args> [--device=<device_index>] "
        "[--instrumented] [--quick] [--v]\n"
        "[--queue-sizing=<scale factor>] [--in-sizing=<in/out queue scale factor>] [--disable-size-check]"
        "[--grid-sizing=<grid size>] [--partition_method=random / biasrandom / clustered / metis] [--partition_seed=<seed>]"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
        "  --instrumented If set then kernels keep track of queue-search_depth\n"
        "  and barrier duty (a relative indicator of load imbalance.)\n"
        "  --quick If set will skip the CPU validation code.\n"
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
 void DisplaySolution(VertexId *comp_ids, SizeT nodes, unsigned int num_components, VertexId *roots, unsigned int *histogram)
 {
    typedef CcList<VertexId> CcListType;
    printf("Number of components: %d\n", num_components);

    if (nodes <= 40) {
        printf("[");
        for (VertexId i = 0; i < nodes; ++i) {
            PrintValue(i);
            printf(":");
            PrintValue(comp_ids[i]);
            printf(",");
            printf(" ");
        }
        printf("]\n");
    }
    else {
        //sort the components by size
        CcListType *cclist = (CcListType*)malloc(sizeof(CcListType) * num_components);
        for (int i = 0; i < num_components; ++i)
        {
            cclist[i].root = roots[i];
            cclist[i].histogram = histogram[i];
        }
        std::stable_sort(cclist, cclist + num_components, CCCompare<CcListType>);

        // Print out at most top 10 largest components
        int top = (num_components < 10) ? num_components : 10;
        printf("Top %d largest components:\n", top);
        for (int i = 0; i < top; ++i)
        {
            printf("CC ID: %d, CC Root: %d, CC Size: %d\n", i, cclist[i].root, cclist[i].histogram);
        }

        free(cclist);
    }
 }

 /**
  * Performance/Evaluation statistics
  */

/******************************************************************************
 * CC Testing Routines
 *****************************************************************************/

/**
 * @brief CPU-based reference CC algorithm using Boost Graph Library
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] row_offsets Host-side vector stores row offsets for each node in the graph
 * @param[in] column_indices Host-side vector stores column indices for each edge in the graph
 * @param[in] num_nodes
 * @param[out] labels Host-side vector to store the component id for each node in the graph
 *
 * \return Number of connected components in the graph
 */
template<
    typename VertexId,
    typename Value, 
    typename SizeT>
unsigned int RefCPUCC(
    const Csr<VertexId, Value, SizeT> &graph,
    int *labels)
{
    using namespace boost;
    SizeT    *row_offsets    = graph.row_offsets;
    VertexId *column_indices = graph.column_indices;
    SizeT     num_nodes      = graph.nodes;

    typedef adjacency_list <vecS, vecS, undirectedS> Graph;
    Graph G;
    for (int i = 0; i < num_nodes; ++i)
    {
        for (int j = row_offsets[i]; j < row_offsets[i+1]; ++j)
        {
            add_edge(i, column_indices[j], G);
        }
    }
    CpuTimer cpu_timer;
    cpu_timer.Start();
    int num_components = connected_components(G, &labels[0]);
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    printf("CPU CC finished in %lf msec.\n", elapsed);
    return num_components;
}

template <
    typename VertexId,
    typename SizeT>
void ConvertIDs(
    VertexId *labels,
    SizeT    num_nodes,
    SizeT    num_components)
{
    VertexId *min_nodes = new VertexId[num_nodes];

    for (int cc=0;cc<num_nodes;cc++)
        min_nodes[cc]=num_nodes;
    for (int node=0;node<num_nodes;node++)
        if (min_nodes[labels[node]]>node) min_nodes[labels[node]]=node;
    for (int node=0;node<num_nodes;node++)
        labels[node]=min_nodes[labels[node]];
    delete[] min_nodes; min_nodes = NULL;
}

/**
 * @brief Run tests for connected component algorithm
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy for CC kernels
 * @param[in] num_gpus Number of GPUs
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
    typedef CCProblem<
        VertexId,
        SizeT,
        Value,
        true> CcProblem; //use double buffer for edgemap and vertexmap.

    typedef CCEnactor<
        CcProblem,
        INSTRUMENT,
        DEBUG,
        SIZE_CHECK> CcEnactor;

    Csr<VertexId, Value, SizeT>
                 *graph                 = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    //VertexId      src                   = (VertexId)parameter -> src;
    int           max_grid_size         = parameter -> max_grid_size;
    int           num_gpus              = parameter -> num_gpus;
    double        max_queue_sizing      = parameter -> max_queue_sizing;
    double        max_in_sizing         = parameter -> max_in_sizing;
    ContextPtr   *context               = (ContextPtr*)parameter -> context;
    std::string   partition_method      = parameter -> partition_method;
    int          *gpu_idx               = parameter -> gpu_idx;
    cudaStream_t *streams               = parameter -> streams;
    float         partition_factor      = parameter -> partition_factor;
    int           partition_seed        = parameter -> partition_seed;
    bool          g_quick               = parameter -> g_quick;
    bool          g_stream_from_host    = parameter -> g_stream_from_host;
    //std::string   ref_filename          = parameter -> ref_filename;
    //int           iterations            = parameter -> iterations;
    size_t       *org_size              = new size_t  [num_gpus];

    // Allocate host-side label array (for both reference and gpu-computed results)
    VertexId    *reference_component_ids= new VertexId[graph->nodes];
    VertexId    *h_component_ids        = new VertexId[graph->nodes];
    VertexId    *reference_check        = (g_quick) ? NULL : reference_component_ids;
    unsigned int ref_num_components     = 0;

    for (int gpu=0; gpu<num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }
    // Allocate CC enactor map
    CcEnactor* enactor = new CcEnactor(num_gpus, gpu_idx);

    // Allocate problem on GPU
    CcProblem* problem = new CcProblem;
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
        partition_seed), "CC Problem Initialization Failed", __FILE__, __LINE__);
    util::GRError(enactor->Init(context, problem, max_grid_size), "BC Enactor Init failed", __FILE__, __LINE__);

    //
    // Compute reference CPU BFS solution for source-distance
    //
    if (reference_check != NULL)
    {
        printf("compute ref value\n");
        ref_num_components = RefCPUCC(
            *graph,
            reference_check);
        printf("\n");
    }

    // Perform CC
    CpuTimer cpu_timer;

    util::GRError(problem->Reset(enactor->GetFrontierType(), max_queue_sizing), "CC Problem Data Reset Failed", __FILE__, __LINE__);
    cpu_timer.Start();
    util::GRError(enactor->Enact(), "CC Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(problem->Extract(h_component_ids), "CC Problem Data Extraction Failed", __FILE__, __LINE__);

    // Validity
    if (ref_num_components == problem->num_components)
    {
        printf("CORRECT.\n");
    } else
        printf("INCORRECT. Ref Component Count: %d, GPU Computed Component Count: %d\n", ref_num_components, problem->num_components);
    if (reference_check != NULL)
    {
        ConvertIDs<VertexId, SizeT>(reference_check, graph->nodes, ref_num_components);
        ConvertIDs<VertexId, SizeT>(h_component_ids, graph->nodes, problem->num_components);
        printf("Label Validity: ");
        int error_num = CompareResults(h_component_ids, reference_check, graph->nodes, true);
        if (error_num>0)
            printf("%d errors occurred.\n", error_num);
        else printf("\n"); 
    }

    //if (ref_num_components == csr_problem->num_components)
    {
        // Compute size and root of each component
        VertexId        *h_roots            = new VertexId    [problem->num_components];
        unsigned int    *h_histograms       = new unsigned int[problem->num_components];

        printf("num_components = %d\n", problem->num_components);
        problem->ComputeCCHistogram(h_component_ids, h_roots, h_histograms);
        printf("num_components = %d\n", problem->num_components);

        // Display Solution
        DisplaySolution(h_component_ids, graph->nodes, problem->num_components, h_roots, h_histograms);
        
        if (h_roots     ) {delete[] h_roots     ; h_roots     =NULL;}
        if (h_histograms) {delete[] h_histograms; h_histograms=NULL;}
    }

    printf("GPU Connected Component finished in %lf msec.\n", elapsed);

    printf("\n\tMemory Usage(B)\t");
    for (int gpu=0;gpu<num_gpus;gpu++)
    if (num_gpus>1)
    {
        if (gpu!=0) printf(" #keys%d\t #ins%d,0\t #ins%d,1",gpu,gpu,gpu);
        else printf(" $keys%d", gpu);
    } else printf(" #keys%d", gpu);
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
            SizeT x=problem->data_slices[gpu]->frontier_queues[i].keys[0].GetSize();
            printf("\t %d", x);
            double factor = 1.0*x/(num_gpus>1?problem->graph_slices[gpu]->in_counter[i]:problem->graph_slices[gpu]->nodes);
            if (factor > max_key_sizing) max_key_sizing=factor;
            if (num_gpus>1 && i!=0 )
            for (int t=0;t<2;t++)
            {
                x=problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                printf("\t %d", x);
                factor = 1.0*x/problem->graph_slices[gpu]->in_counter[i];
                if (factor > max_in_sizing_) max_in_sizing_=factor;
            }
        }
        if (num_gpus>1) printf("\t %d",problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize());
        printf("\n");
    }
    printf("\t key_sizing =\t %lf", max_key_sizing);
    if (num_gpus>1) printf("\t in_sizing =\t %lf", max_in_sizing_);
    printf("\n");

    // Cleanup 
    if (org_size               ) {delete[] org_size               ; org_size                = NULL;}
    if (problem                ) {delete   problem                ; problem                 = NULL;}
    if (enactor                ) {delete   enactor                ; enactor                 = NULL;}
    if (reference_component_ids) {delete[] reference_component_ids; reference_component_ids = NULL;}
    if (h_component_ids        ) {delete[] h_component_ids        ; h_component_ids         = NULL;}

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

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT>
void RunTests_instrumented(Test_Parameter *parameter)
{
    if (parameter->instrumented) RunTests_debug
        <VertexId, Value, SizeT,
        true > (parameter);
    else RunTests_debug
        <VertexId, Value, SizeT,
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
void RunTests(
    Csr<VertexId, Value, SizeT> *graph,
    CommandLineArgs             &args,
    int                          num_gpus,
    ContextPtr                  *context,
    int                         *gpu_idx,
    cudaStream_t                *streams)
{
    Test_Parameter *parameter = new Test_Parameter;
    parameter -> Init(args);
    parameter -> graph        = graph;
    parameter -> num_gpus     = num_gpus;
    parameter -> context      = context;
    parameter -> gpu_idx      = gpu_idx;
    parameter -> streams      = streams;

    RunTests_instrumented<VertexId, Value, SizeT>(parameter);
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
    bool             g_undirected = false; //Does not make undirected graph

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

	if (graph_type == "market") {

		// Matrix-market coordinate-formatted graph file

		typedef int VertexId;							// Use as the node identifier type
		typedef int Value;								// Use as the value type
		typedef int SizeT;								// Use as the graph size type
		Csr<VertexId, Value, SizeT> csr(false);         // default value for stream_from_host is false

        printf("size of int:%ld\n", sizeof(int));

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
        fflush(stdout);

		// Run tests
		RunTests(&csr, args, num_gpus, context, gpu_idx, streams);

	} else {

		// Unknown graph type
		fprintf(stderr, "Unspecified graph type\n");
		return 1;

	}

	return 0;
}
