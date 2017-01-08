// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * test_tc.cu
 *
 * @brief Simple test driver for computing TC.
 */

#include <stdio.h>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utilities
#include <gunrock/graphio/market.cuh>

// MST includes
#include <gunrock/global_indicator/tc/tc_enactor.cuh>
#include <gunrock/global_indicator/tc/tc_problem.cuh>
#include <gunrock/global_indicator/tc/tc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/intersection/kernel.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::global_indicator::tc;

///////////////////////////////////////////////////////////////////////////////
// Housekeeping and utility routines
///////////////////////////////////////////////////////////////////////////////

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
        "[--quick]                 Skip the CPU reference validation process.\n"
        "[--disable-size-check]    Disable frontier queue size check.\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

///////////////////////////////////////////////////////////////////////////////
// GPU TC test routines
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Test entry
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
    //bool     DEBUG,
    //bool     SIZE_CHECK >
cudaError_t RunTest(Info<VertexId, Value, SizeT> *info)
{
    // define the problem data structure for graph primitive
    typedef TCProblem<VertexId,
            SizeT,
            Value>
            Problem;
    typedef TCEnactor <Problem>
            Enactor;

    Csr<VertexId, SizeT, Value>* graph = info->csr_ptr;

    int     num_gpus                = 1; //info->info["num_gpus"         ].get_int ();
    int     max_grid_size           = info->info["max_grid_size"    ].get_int ();
    int     iterations              = 1; //force to 1 info->info["num_iteration"].get_int();
    bool    quiet_mode              = info->info["quiet_mode"       ].get_bool ();
    bool    quick_mode              = info->info["quick_mode"       ].get_bool ();
    bool    instrument              = info->info["instrument"       ].get_bool (); 
    bool    debug                   = info->info["debug_mode"       ].get_bool (); 
    bool    size_check              = info->info["size_check"       ].get_bool (); 
    bool    stream_from_host        = info->info["stream_from_host" ].get_bool ();
    double  max_queue_sizing        = info->info["max_queue_sizing" ].get_real ();
    double  max_queue_sizing1       = info->info["max_queue_sizing1"].get_real ();
    double  max_in_sizing           = info->info["max_in_sizing"    ].get_real (); 
    std::string partition_method    = info->info["partition_method" ].get_str  (); 
    double   partition_factor       = info->info["partition_factor" ].get_real (); 
    int      partition_seed         = info->info["partition_seed"   ].get_int  ();
    CpuTimer cpu_timer; 
    cudaError_t retval             = cudaSuccess;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) 
        gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*  )info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;


    VertexId *h_srcs              = new VertexId[graph->edges/2];
    VertexId *h_dsts              = new VertexId[graph->edges/2];
    SizeT    *h_tc                = new SizeT[graph->edges/2];

    printf("size: %d\n", graph->edges/2);

    // allocate problem on GPU create a pointer of the MSTProblem type
    Problem problem(true);

    // host results spaces
    if (!quiet_mode) { printf("\nTC TEST\n"); }

    // copy data from CPU to GPU initialize data members in DataSlice
    util::GRError(problem.Init(stream_from_host,
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
                  "Problem TC Initialization Failed", __FILE__, __LINE__);

    // allocate TC enactor map
    Enactor enactor(
    num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor.Init(
    context, &problem, max_grid_size));
    cpu_timer.Stop();
    info->info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // perform calculations
    //GpuTimer gpu_timer;  // record the kernel running time
    double elapsed_gpu = 0.0f;  // device elapsed running time

    for (int iter = 0; iter < iterations; ++iter)
    {
        // reset values in DataSlice
        util::GRError(problem.Reset(
                          enactor.GetFrontierType(),
                          max_queue_sizing,
                          max_queue_sizing1),
                      "TC Problem Data Reset Failed", __FILE__, __LINE__);

        util::GRError(enactor.Reset(),
            "TC Enactor Reset failed", __FILE__, __LINE__);

        cpu_timer.Start();

        // launch TC enactor
        util::GRError(enactor.template Enact<30>(),
                      "TC Problem Enact Failed", __FILE__, __LINE__);

        cpu_timer.Stop();
        elapsed_gpu += cpu_timer.ElapsedMillis();
    }

    elapsed_gpu /= iterations;
    cpu_timer.Start();

    if (!quiet_mode)
    {
        printf("GPU - Computation Complete in %lf msec.\n", elapsed_gpu);
    }

    // copy out results
    util::GRError(problem.Extract(h_srcs, h_dsts, h_tc),
                  "TC Problem Data Extraction Failed", __FILE__, __LINE__);

    // write to mtx file
    std::ofstream fout("tc_weight_graph.mtx");
    if (fout.is_open())
    {
        fout << graph->nodes << " " << graph->nodes << " " << graph->edges/2 << std::endl;
        for (int i = 0; i < graph->edges/2; ++i)
            fout << h_srcs[i]+1 << " " << h_dsts[i]+1 << " " << h_tc[i] << std::endl;
        fout.close();
    }

    info->ComputeCommonStats(enactor.enactor_stats.GetPointer(), elapsed_gpu, (VertexId*)NULL);

    // clean up if necessary
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    return retval;
    
}

///////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{

    cudaError_t retval = cudaSuccess;
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

    CpuTimer cpu_timer, cpu_timer2;
    cpu_timer.Start();

    Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
    Info<VertexId, Value, SizeT> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    info->info["undirected"] = true;  // always convert to undirected

    cpu_timer2.Start();
    info->Init("TC", args, csr);
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    retval = RunTest<VertexId, SizeT, Value>(info);

    cpu_timer.Stop();
    info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!(info->info["quiet_mode"].get_bool()))
    {
        info->DisplayStats();
    }

    info->CollectInfo();
    
    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
