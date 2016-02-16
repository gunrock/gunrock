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
    typename Value,
    bool     DEBUG,
    bool     SIZE_CHECK >
void RunTest(Info<VertexId, Value, SizeT> *info)
{
    // define the problem data structure for graph primitive
    typedef TCProblem<VertexId,
            SizeT,
            Value>
            Problem;

    Csr<VertexId, Value, SizeT>* graph =
        (Csr<VertexId, Value, SizeT>*)info->csr_ptr;

    int num_gpus            = info->info["num_gpus"].get_int();
    int max_grid_size       = info->info["max_grid_size"].get_int();
    int iterations          = 1; //force to 1 info->info["num_iteration"].get_int();
    bool quiet_mode         = info->info["quiet_mode"].get_bool();
    bool quick_mode         = info->info["quick_mode"].get_bool();
    bool stream_from_host   = info->info["stream_from_host"].get_bool();
    double max_queue_sizing = info->info["max_queue_sizing"].get_real();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr* context = (ContextPtr*)info->context;

    // allocate MST enactor map
    TCEnactor < Problem,
               false,        // INSTRUMENT
               DEBUG,        // DEBUG
               SIZE_CHECK >  // SIZE_CHECK
               enactor(gpu_idx);

    // allocate problem on GPU create a pointer of the MSTProblem type
    Problem * problem = new Problem;

    // host results spaces
    VertexId * edge_mask = new VertexId[graph->edges];

    if (!quiet_mode) { printf("\nTC TEST\n"); }

    // copy data from CPU to GPU initialize data members in DataSlice
    util::GRError(problem->Init(stream_from_host, *graph, num_gpus),
                  "Problem TC Initialization Failed", __FILE__, __LINE__);

    // perform calculations
    GpuTimer gpu_timer;  // record the kernel running time
    double elapsed_gpu = 0.0f;  // device elapsed running time

    for (int iter = 0; iter < iterations; ++iter)
    {
        // reset values in DataSlice
        util::GRError(problem->Reset(
                          enactor.GetFrontierType(), max_queue_sizing),
                      "TC Problem Data Reset Failed", __FILE__, __LINE__);

        gpu_timer.Start();

        // launch MST enactor
        util::GRError(enactor.template Enact<Problem>(
                          *context, problem, max_grid_size),
                      "TC Problem Enact Failed", __FILE__, __LINE__);

        gpu_timer.Stop();
        elapsed_gpu += gpu_timer.ElapsedMillis();
    }

    elapsed_gpu /= iterations;
    if (!quiet_mode)
    {
        printf("GPU - Computation Complete in %lf msec.\n", elapsed_gpu);
    }

    info->ComputeCommonStats(enactor.enactor_stats.GetPointer(), elapsed_gpu);

    info->CollectInfo();

    // clean up if necessary
    if (problem)   delete    problem;
}

/**
 * @brief Test entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam DEBUG
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool     DEBUG >
void RunTests_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
    {
        RunTest <VertexId, Value, SizeT, DEBUG,  true>(info);
    }
    else
    {
        RunTest <VertexId, Value, SizeT, DEBUG, false>(info);
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
    typename VertexId,
    typename Value,
    typename SizeT >
void RunTests_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
    {
        RunTests_size_check <VertexId, Value, SizeT,  true>(info);
    }
    else
    {
        RunTests_size_check <VertexId, Value, SizeT, false>(info);
    }
}


///////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////

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
    info->info["undirected"] = true;  // always convert to undirected
    info->Init("TC", args, csr);
    RunTests_debug(info);

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
