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
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::cc;

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
        "[--partition_method=<random|biasrandom|clustered|metis>]\n"
        "                          Choose partitioner (Default use random).\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
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
void DisplaySolution(
    VertexId     *comp_ids,
    SizeT        nodes,
    unsigned int num_components,
    VertexId     *roots,
    unsigned int *histogram)
{
    typedef CcList<VertexId> CcListType;
    //printf("Number of components: %d\n", num_components);

    if (nodes <= 40)
    {
        printf("[");
        for (VertexId i = 0; i < nodes; ++i)
        {
            PrintValue(i);
            printf(":");
            PrintValue(comp_ids[i]);
            printf(",");
            printf(" ");
        }
        printf("]\n");
    }
    else
    {
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

        free(cclist);
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
void RunTests(Info<VertexId, Value, SizeT> *info)
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

    //if (ref_num_components == csr_problem->num_components)
    {
        // Compute size and root of each component
        VertexId     *h_roots      = new VertexId    [problem->num_components];
        unsigned int *h_histograms = new unsigned int[problem->num_components];

        //printf("num_components = %d\n", problem->num_components);
        problem->ComputeCCHistogram(h_component_ids, h_roots, h_histograms);
        //printf("num_components = %d\n", problem->num_components);

        if (!quiet_mode)
        {
            // Display Solution
            DisplaySolution(h_component_ids, graph->nodes,
                            problem->num_components, h_roots, h_histograms);
        }

        if (h_roots     ) {delete[] h_roots     ; h_roots      = NULL;}
        if (h_histograms) {delete[] h_histograms; h_histograms = NULL;}
    }

    info->ComputeCommonStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), elapsed, h_component_ids, true);

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
        RunTests<VertexId, Value, SizeT, INSTRUMENT, DEBUG,  true>(info);
    }
    else
    {
        RunTests<VertexId, Value, SizeT, INSTRUMENT, DEBUG, false>(info);
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

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    typedef int VertexId;  // use int as the vertex identifier
    typedef int Value;     // use int as the value type
    typedef int SizeT;     // use int as the graph size type

    Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
    Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info->info["undirected"] = true;   // require undirected input graph

    info->Init("CC", args, csr);  // initialize Info structure
    graphio::RemoveStandaloneNodes<VertexId, Value, SizeT>(
        &csr, args.CheckCmdLineFlag("quiet"));
    RunTests_instrumented<VertexId, Value, SizeT>(info);  // run test

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
