// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file test_sample.cu
 *
 * @brief Simple test driver program
 */

#include <stdio.h>
#include <string>
#include <iostream>

// utilities for correctness checking
#include <gunrock/util/test_utils.cuh>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// primitive-specific headers include
#include <gunrock/app/template/sample_enactor.cuh>
#include <gunrock/app/template/sample_problem.cuh>
#include <gunrock/app/template/sample_functor.cuh>

// gunrock abstraction graph operators
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sample;

// ----------------------------------------------------------------------------
// Housekeeping Routines
// ----------------------------------------------------------------------------
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
        "[--src=<Vertex-ID|randomize|largestdegree>]\n"
        "                          Begins traversal from the source (Default: 0).\n"
        "                          If randomize: from a random source vertex.\n"
        "                          If largestdegree: from largest degree vertex.\n"
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
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays primitive results.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph.
 */
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(const Csr<VertexId, SizeT, Value> &graph)
{
    printf("==> display solution: (currently missing)\n");
    // TODO(developer): code to print out results
}

// ----------------------------------------------------------------------------
// Testing Routines
// ----------------------------------------------------------------------------

/**
 * @brief A simple CPU-based reference implementation.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph Reference to the CSR graph we process on.
 */
template<typename VertexId, typename SizeT, typename Value>
void ReferenceSample(const Csr<VertexId, SizeT, Value> &graph)
{
    // initialization

    // perform calculation

    CpuTimer cpu_timer;
    cpu_timer.Start();

    // TODO(developer): CPU validation code here

    cpu_timer.Stop();

    float cpu_elapsed = cpu_timer.ElapsedMillis();
    printf("CPU reference finished in %lf ms.\n\n", cpu_elapsed);
}

/**
 * @brief Sample test entry
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
    //bool DEBUG,
    //bool SIZE_CHECK >
void RunTest(Info<VertexId, SizeT, Value> *info)
{
    typedef SampleProblem < VertexId, SizeT, Value,
            true,   // MARK_PREDECESSORS
            false>  // ENABLE_IDEMPOTENCE
            //false > 
            Problem;
    typedef SampleEnactor <Problem>
            Enactor;

    Csr<VertexId, SizeT, Value>* csr = info->csr_ptr;
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
    int      iterations            = 1; //disable since doesn't support mgpu stop condition. info->info["num_iteration"].get_int();
    CpuTimer cpu_timer;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // allocate host-side array (for both reference and GPU-computed results)
    VertexId *r_labels = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);
    VertexId *h_labels = (VertexId*)malloc(sizeof(VertexId) * csr->nodes);

    Problem *problem = new Problem(false);  // allocate primitive problem on GPU
    util::GRError(problem -> Init(
        stream_from_host,
        csr,
        NULL,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "Problem Initialization Failed", __FILE__, __LINE__);

    // allocate primitive enactor map
    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor->Init(
        context, problem, max_grid_size),
        "Enactor Init failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();
   
    //
    // perform calculation
    //
    double elapsed = 0.0;

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem -> Reset(
            enactor -> GetFrontierType(),
            max_queue_sizing, max_queue_sizing1),
            "Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(enactor -> Reset(),
            "Enactor Reset failed", __FILE__, __LINE__);

        cpu_timer.Start();
        util::GRError(enactor -> Enact(),
            "Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        elapsed += cpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    cpu_timer.Start();
    // extract results
    util::GRError(
        problem->Extract(h_labels),
        "Problem Data Extraction Failed", __FILE__, __LINE__);

    // compute reference CPU validation solution
    if (!quick_mode)
    {
        if (!quiet_mode) printf("==> computing reference value ... (currently missing)\n");
        ReferenceSample<VertexId, SizeT, Value>(csr);
        if (!quiet_mode) printf("==> validation: (currently missing)\n");
    }

    if (!quiet_mode) DisplaySolution<VertexId, SizeT, Value>(csr);  // display solution

    info->ComputeCommonStats(enactor -> enactor_stats.GetPointer(), elapsed, (VertexId*)NULL);

    // clean up
    if (problem ) { delete problem; }
    if (enactor ) { delete enactor; }
    if (r_labels) { free(r_labels); }
    if (h_labels) { free(h_labels); }
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
template <
    typename VertexId,  // use int as the vertex identifier
    typename SizeT   ,  // use int as the graph size type
    typename Value   >  // use int as the value type
int main_(CommandLineArgs *args)
{
    CpuTimer cpu_timer, cpu_timer2;
    cpu_timer.Start();

    Csr <VertexId, SizeT, Value> csr(false);  // graph we process on
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    info->info["undirected"] = args -> CheckCmdLineFlag("undirected");
    cpu_timer2.Start();
    info->Init("Primitive_Name", *args, csr);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    RunTest<VertexId, SizeT, Value>(info);  // run test

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
// disabled to reduce compile time
//    if (args -> CheckCmdLineFlag("64bit-Value"))
//        return main_<VertexId, SizeT, long long>(args);
//    else
        return main_<VertexId, SizeT, int      >(args);
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
