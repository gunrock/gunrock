// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * test_sm.cu
 *
 * @brief Simple test driver for computing Minimum Spanning Tree.
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

// SM includes
#include <gunrock/app/sm/sm_enactor.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/app/sm/sm_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sm;

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

/**
 * @brief Displays the SM result.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph.
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(const Csr<VertexId, Value, SizeT> &graph_query,
                     const Csr<VertexId, Value, SizeT> &graph_data,
                     VertexId *h_froms,
                     VertexId *h_tos,
                     SizeT num_matches)
{
    // TODO(developer): code to print out results
    printf("Number of matched subgraphs: %u.\n",num_matches);

}

///////////////////////////////////////////////////////////////////////////////
// CPU validation routines
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief A simple CPU-based reference subgraph matching implementation.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 *
 * @param[in] graph_query Reference to the CSR query graph we process on.
 * @param[in] graph_data  Referece to the CSR data graph we process on.
 * @param[in] h_c_set     Reference to the candidate set matrix result.
 */
template<typename VertexId, typename Value, typename SizeT>
Value SimpleReferenceSM(
	const Csr<VertexId, Value, SizeT> &graph_query,
        const Csr<VertexId, Value, SizeT> &graph_data,
	bool quiet_mode = false)
{
    if (!quiet_mode) { printf("\nSM CPU REFERENCE TEST\n"); }

    SizeT num_matches = 0;

    CpuTimer cpu_timer; // record the kernel running time
    cpu_timer.Start();

    // TODO(developer): CPU validation code here

    cpu_timer.Stop();
    float elapsed_cpu = cpu_timer.ElapsedMillis();

    if (!quiet_mode)
    {
        printf("CPU - Computation Complete in %lf msec.\n", elapsed_cpu);
        // printf("CPU - Number of Edges in MST: %d\n", num_selected_cpu);
    }
    return num_matches;
}

///////////////////////////////////////////////////////////////////////////////
// GPU SM test routines
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
    typedef SMProblem<VertexId,
            SizeT,
            Value,
            true,    // MARK_PREDECESSORS
            false,   // ENABLE_IDEMPOTENCE
            true >   // USE_DOUBLE_BUFFER
            Problem;

    Csr<VertexId, Value, SizeT>* graph_query =
        (Csr<VertexId, Value, SizeT>*)info->csr_query_ptr;
    Csr<VertexId, Value, SizeT>* graph_data =
        (Csr<VertexId, Value, SizeT>*)info->csr_data_ptr;

    int num_gpus            = info->info["num_gpus"].get_int();
    int max_grid_size       = info->info["max_grid_size"].get_int();
    int iterations          = 1; //force to 1 info->info["num_iteration"].get_int();
    bool quiet_mode         = info->info["quiet_mode"].get_bool();
    bool quick_mode         = info->info["quick_mode"].get_bool();
    bool stream_from_host   = info->info["stream_from_host"].get_bool();
    double max_queue_sizing = info->info["max_queue_sizing"].get_real();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) 
        gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr* context = (ContextPtr*)info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // allocate SM enactor map
    SMEnactor < Problem,
               false,        // INSTRUMENT
               DEBUG,        // DEBUG
               SIZE_CHECK >  // SIZE_CHECK
               enactor(gpu_idx);

    // allocate problem on GPU create a pointer of the MSTProblem type
    Problem * problem = new Problem;

    // host results spaces
    VertexId *h_froms = new VertexId[1000];
    VertexId *h_tos = new VertexId[1000];

    if (!quiet_mode) { printf("\nSUBGRAPH MATCHING TEST\n"); fflush(stdout);}

    // copy data from CPU to GPU initialize data members in DataSlice
    util::GRError(problem->Init(stream_from_host, *graph_query, *graph_data, num_gpus, gpu_idx, streams),
                  "Problem SM Initialization Failed", __FILE__, __LINE__);
    // perform calculations
    GpuTimer gpu_timer;  // record the kernel running time
    double elapsed_gpu = 0.0f;  // device elapsed running time

    for (int iter = 0; iter < iterations; ++iter)
    {
        // reset values in DataSlice
        util::GRError(problem->Reset(
                          enactor.GetFrontierType(), max_queue_sizing),
                      "SM Problem Data Reset Failed", __FILE__, __LINE__);

        gpu_timer.Start();

        // launch SM enactor
        util::GRError(enactor.template Enact<Problem>(
                          *context, problem, max_grid_size),
                      "SM Problem Enact Failed", __FILE__, __LINE__);

        gpu_timer.Stop();
        elapsed_gpu += gpu_timer.ElapsedMillis();
    }

    elapsed_gpu /= iterations;
    if (!quiet_mode)
    {
        printf("GPU - Computation Complete in %lf msec.\n", elapsed_gpu);
    }

    // copy results back to CPU from GPU using Extract
    util::GRError(problem->Extract(h_froms, h_tos),
                  "SM Problem Data Extraction Failed", __FILE__, __LINE__);
    SizeT num_matches_gpu = problem->data_slices[0]->num_matches;

    if (!quick_mode)  // run CPU reference test
    {
        // correctness validation
	SizeT num_matches_cpu =  SimpleReferenceSM(*graph_query, *graph_data, quiet_mode);
	if(num_matches_cpu == num_matches_gpu)
	{
            if (!quiet_mode) DisplaySolution(*graph_query, *graph_data, h_froms, h_tos, num_matches_cpu);
            if (!quiet_mode) { printf("\nCORRECT.\n"); }
        }
        else
        {
            if (!quiet_mode)
            {
                printf("INCORRECT.\n");
                std::cout << "CPU Number of matched subgraphs = " << num_matches_cpu << std::endl;
                std::cout << "GPU Number of matched subgraphs = " << num_matches_gpu << std::endl;
            }
        }
    }


    info->ComputeCommonStats(enactor.enactor_stats.GetPointer(), elapsed_gpu);

    if (!quiet_mode)
    {
        info->DisplayStats(false);   // display collected statistics
    }

    info->CollectInfo();

    // clean up if necessary
    if (problem)   delete    problem;
    if (h_froms) delete [] h_froms;
    if (h_tos) delete [] h_tos;
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
    if (argc < 3 || graph_args < 2 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    typedef int VertexId;  // use int as the vertex identifier
    typedef int Value;     // use int as the value type
    typedef int SizeT;     // use int as the graph size type

    Csr<VertexId, Value, SizeT> csr_query(false);  // query graph we process on
    Csr<VertexId, Value, SizeT> csr_data(false);  // data graph we process on

    Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info->info["undirected"] = true;  // always convert to undirected
    //info->info["debug_mode"] = true;  // debug mode

    if(graph_args == 5)  info->info["node_value"] = true;  // require per node label values

    info->Init("SM", args, csr_query, csr_data);

    graphio::RemoveStandaloneNodes<VertexId, Value, SizeT>(
        &csr_data, args.CheckCmdLineFlag("quite"));

    RunTests_debug<VertexId, Value, SizeT>(info);  // run test

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
