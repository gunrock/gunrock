// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_topk.cu
 *
 * @brief Simple test driver program for computing Topk.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <map>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Degree Centrality includes
#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::topk;

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
        "[--undirected]            Treat the graph as undirected (symmetric).\n"
        "[--instrumented]          Keep kernels statics [Default: Disable].\n"
        "                          total_queued, search_depth and barrier duty.\n"
        "                          (a relative indicator of load imbalance.)\n"
        "[--quick]                 Skip the CPU reference validation process.\n"
        "[--disable-size-check]    Disable frontier queue size check.\n"
        "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
        "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief displays the top K results
 *
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
void DisplaySolution(
    VertexId *h_node_id,
    Value    *h_degrees_i,
    Value    *h_degrees_o,
    SizeT    num_nodes)
{
    fflush(stdout);
    // at most display the first 100 results
    if (num_nodes > 100) num_nodes = 100;
    printf("==> top %lld centrality nodes:\n", 
        (long long)num_nodes);
    for (SizeT iter = 0; iter < num_nodes; ++iter)
        printf("%lld %lld %lld\n",
            (long long)h_node_id  [iter], 
            (long long)h_degrees_i[iter], 
            (long long)h_degrees_o[iter]);
}

/******************************************************************************
 * Degree Centrality Testing Routines
 *****************************************************************************/
/**
 * @brief A simple CPU-based reference TOPK implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 */
struct compare_second_only
{
    template <typename T1, typename T2>
    bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2)
    {
        return p1.second > p2. second;
    }
};

template <
    typename VertexId,
    typename SizeT,
    typename Value>
void ReferenceTopK(
    Csr<VertexId, SizeT, Value> &csr,
    Csr<VertexId, SizeT, Value> &csc,
    VertexId *ref_node_id,
    SizeT    *ref_degrees,
    SizeT    top_nodes)
{
    printf("CPU reference test.\n");
    CpuTimer cpu_timer;

    // malloc degree centrality spaces
    Value *ref_degrees_original =
        (Value*)malloc(sizeof(Value) * csr.nodes);
    Value *ref_degrees_reversed =
        (Value*)malloc(sizeof(Value) * csc.nodes);

    // store reference output results
    std::vector< std::pair<VertexId, Value> > results;

    // calculations
    for (SizeT node = 0; node < csr.nodes; ++node)
    {
        ref_degrees_original[node] =
            csr.row_offsets[node + 1] - csr.row_offsets[node];
        ref_degrees_reversed[node] =
            csc.row_offsets[node + 1] - csc.row_offsets[node];
    }

    cpu_timer.Start();

    // add ingoing degrees and outgoing degrees together
    for (SizeT node = 0; node < csr.nodes; ++node)
    {
        ref_degrees_original[node] =
            ref_degrees_original[node] + ref_degrees_reversed[node];
        results.push_back( std::make_pair (node, ref_degrees_original[node]) );
    }

    // pair sort according to second elements - degree centrality
    std::stable_sort(results.begin(), results.end(), compare_second_only());

    for (SizeT itr = 0; itr < top_nodes; ++itr)
    {
        ref_node_id[itr] = results[itr].first;
        ref_degrees[itr] = results[itr].second;
    }

    cpu_timer.Stop();
    float elapsed_cpu = cpu_timer.ElapsedMillis();
    printf("==> CPU Degree Centrality finished in %lf msec.\n", elapsed_cpu);

    // clean up if neccessary
    if (ref_degrees_original) { free(ref_degrees_original); }
    if (ref_degrees_reversed) { free(ref_degrees_reversed); }
    results.clear();
}

/**
 * @brief Run TopK tests
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
    typename SizeT,
    typename Value>
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK >
void RunTests(Info<VertexId, SizeT, Value> *info)
{
    // define the problem data structure for graph primitive
    typedef TOPKProblem <
        VertexId,
        SizeT,
        Value > Problem;
    typedef TOPKEnactor <Problem>
        Enactor;

    Csr<VertexId, SizeT, Value> *csr = info->csr_ptr;
    Csr<VertexId, SizeT, Value> *csc = info->csc_ptr;
    int      max_grid_size         = info->info["max_grid_size"     ].get_int  (); 
    int      num_gpus              = info->info["num_gpus"          ].get_int  (); 
    double   max_queue_sizing      = info->info["max_queue_sizing"  ].get_real (); 
    double   max_queue_sizing1     = info->info["max_queue_sizing1" ].get_real (); 
    double   max_in_sizing         = info->info["max_in_sizing"     ].get_real (); 
    std::string partition_method   = info->info["partition_method"  ].get_str  (); 
    double   partition_factor      = info->info["partition_factor"  ].get_real (); 
    int      partition_seed        = info->info["partition_seed"    ].get_int  (); 
    bool     quiet_mode            = info->info["quiet_mode"        ].get_bool (); 
    bool     quick_mode            = info->info["quick_mode"        ].get_bool (); 
    bool     stream_from_host      = info->info["stream_from_host"  ].get_bool (); 
    bool     instrument            = info->info["instrument"        ].get_bool (); 
    bool     debug                 = info->info["debug_mode"        ].get_bool (); 
    bool     size_check            = info->info["size_check"        ].get_bool (); 
    SizeT    top_nodes             = info->info["top_nodes"         ].get_int  ();
    CpuTimer cpu_timer;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();
    
    ContextPtr    *context = (ContextPtr*  )info->context;
    cudaStream_t  *streams = (cudaStream_t*)info->streams;

    // reset top_nodes if input k > total number of nodes
    if (top_nodes > csr->nodes) top_nodes = csr->nodes;

    // malloc host memory
    VertexId *h_node_id   = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
    VertexId *ref_node_id = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
    SizeT    *h_degrees_i = (SizeT*)malloc(sizeof(SizeT) * top_nodes);
    SizeT    *h_degrees_o = (SizeT*)malloc(sizeof(SizeT) * top_nodes);
    SizeT    *ref_degrees = (SizeT*)malloc(sizeof(SizeT) * top_nodes);

    // allocate problem on GPU
    // create a pointer of the TOPKProblem type
    Problem *problem = new Problem;

    // copy data from CPU to GPU
    // initialize data members in DataSlice for graph
    util::GRError(problem->Init(
        stream_from_host,
        csr,
        csc,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "Problem TOPK Initialization Failed", __FILE__, __LINE__);

    // instrument specifies whether we want to keep such statistical data
    // Allocate TOPK enactor map
    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor->Init(
        context, problem, max_grid_size),
        "Enactor Init failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // perform topk degree centrality calculations

    // reset values in DataSlice for graph
    util::GRError(problem->Reset(enactor -> GetFrontierType(),
        max_queue_sizing, max_queue_sizing1),
        "TOPK Problem Data Reset Failed", __FILE__, __LINE__);
    util::GRError(enactor -> Reset(),
        "TOPK Enactor Reset failed", __FILE__, __LINE__);

    cpu_timer.Start();
    // launch topk enactor
    util::GRError(enactor -> Enact(top_nodes),
        "TOPK Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();

    double elapsed_gpu = cpu_timer.ElapsedMillis();
    cpu_timer.Start();
    printf("==> GPU TopK Degree Centrality finished in %lf msec.\n", elapsed_gpu);

    // copy out results back to CPU from GPU using Extract
    util::GRError(problem->Extract(
        h_node_id,
        h_degrees_i,
        h_degrees_o,
        top_nodes),
        "TOPK Problem Data Extraction Failed", __FILE__, __LINE__);

    // display solution
    if (!quiet_mode)
        DisplaySolution(
            h_node_id,
            h_degrees_i,
            h_degrees_o,
            top_nodes);

    info->ComputeCommonStats(enactor -> enactor_stats.GetPointer(), elapsed_gpu, (VertexId*)NULL);

    // validation
    ReferenceTopK(
        *csr,
        *csc,
        ref_node_id,
        ref_degrees,
        top_nodes);

    int error_num = CompareResults(h_node_id, ref_node_id, top_nodes, true);
    if (error_num > 0)
    {
        if (!quiet_mode) printf("INCOREECT! %d error(s) occured. \n", error_num);
    }
    if (!quiet_mode) printf("\n");

    // cleanup if neccessary
    if (problem)      { delete problem; }
    if (enactor)      { delete enactor; }
    if (h_node_id)    { free(h_node_id);   }
    if (h_degrees_i)  { free(h_degrees_i); }
    if (h_degrees_o)  { free(h_degrees_o); }
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    //cudaDeviceSynchronize();
}

/******************************************************************************
 * Main
 ******************************************************************************/
template <
    typename VertexId,  // use int as the vertex identifier
    typename SizeT   ,  // use int as the graph size type
    typename Value   >  // use int as the value type
int main_(CommandLineArgs *args)
{
    CpuTimer cpu_timer, cpu_timer2;
    cpu_timer.Start();
    Csr <VertexId, SizeT, Value> csr(false);
    Csr <VertexId, SizeT, Value> csc(false);
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    info->info["undirected"] = args -> CheckCmdLineFlag("undirected");
    cpu_timer2.Start();
    info->Init("TOPK", *args, csr, csc);
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    RunTests<VertexId, SizeT, Value>(info);
    cpu_timer.Stop();
    info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!(info->info["quiet_mode"].get_bool()))
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject

    return 0;
}

template <
    typename VertexId, // the vertex identifier type, usually int or long long
    typename SizeT   > // the size tyep, usually int or long long
int main_Value(CommandLineArgs *args)
{
    // Value = SizeT for TopK
    return main_<VertexId, SizeT, SizeT>(args);
    //if (args -> CheckCmdLineFlag("64bit-Value"))
    //    return main_<VertexId, SizeT, long long>(args);
    //else
    //    return main_<VertexId, SizeT, int      >(args);
}

template <
    typename VertexId>
int main_SizeT(CommandLineArgs *args)
{
// disabled to reduce compile time
//    if (args -> CheckCmdLineFlag("64bit-SizeT"))
//        return main_Value<VertexId, long long>(args);
//    else
        return main_Value<VertexId, int      >(args);
}

int main_VertexId(CommandLineArgs *args)
{
    // disabled, because oprtr::filter::KernelPolicy::SmemStorage is too large for 64bit VertexId
    //if (args -> CheckCmdLineFlag("64bit-VertexId"))
    //    return main_SizeT<long long>(args);
    //else 
        return main_SizeT<int      >(args);
}

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    return main_VertexId(&args);
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
