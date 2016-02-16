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

// CPU reference include
//#include <Escape/GraphIO.h>
//#include <>

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
template<typename VertexId, typename SizeT, typename Value>
void DisplaySolution(
    const Csr<VertexId, SizeT, Value> &graph_query,
    const Csr<VertexId, SizeT, Value> &graph_data,
    VertexId *h_froms,
    VertexId *h_tos,
    SizeT num_matches)
{
    // TODO(developer): code to print out results
    printf("Number of matched subgraphs: %lld.\n",
        (long long)num_matches);

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
template<typename VertexId, typename SizeT, typename Value>
SizeT ReferenceSM(
	const Csr<VertexId, SizeT, Value> &graph_query,
    const Csr<VertexId, SizeT, Value> &graph_data,
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
    typename Value>
    //bool     DEBUG,
    //bool     SIZE_CHECK >
void RunTests(Info<VertexId, SizeT, Value> *info)
{
    // define the problem data structure for graph primitive
    typedef SMProblem<VertexId,
            SizeT,
            Value,
            true,    // MARK_PREDECESSORS
            false>   // ENABLE_IDEMPOTENCE
            //true >   // USE_DOUBLE_BUFFER
            Problem;
    typedef SMEnactor <Problem>
            Enactor;

    Csr<VertexId, SizeT, Value> *graph_query = info->csr_query_ptr;
    Csr<VertexId, SizeT, Value> *graph_data  = info->csr_data_ptr;
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
    for (int i = 0; i < num_gpus; i++) 
        gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*  )info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // host results spaces
    VertexId *h_froms = new VertexId[1000];
    VertexId *h_tos   = new VertexId[1000];

    // allocate problem on GPU create a pointer of the MSTProblem type
    Problem * problem = new Problem(true);
    // copy data from CPU to GPU initialize data members in DataSlice
    util::GRError(problem->Init(
        stream_from_host,
        graph_query,
        graph_data,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "Problem SM Initialization Failed", __FILE__, __LINE__);

    // allocate SM enactor map
    // allocate primitive enactor map
    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor->Init(
        context, problem, max_grid_size),
        "SMEnactor Init failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    if (!quiet_mode) { printf("\nSUBGRAPH MATCHING TEST\n"); fflush(stdout);}

    // perform calculations
    double elapsed_gpu = 0.0f;  // device elapsed running time

    for (int iter = 0; iter < iterations; ++iter)
    {
        // reset values in DataSlice
        util::GRError(problem->Reset(
            enactor -> GetFrontierType(), 
            max_queue_sizing, max_queue_sizing1),
            "SM Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(enactor -> Reset(),
            "Enactor Reset failed", __FILE__, __LINE__);

        cpu_timer.Start();
        // launch SM enactor
        util::GRError(enactor -> Enact(),
            "SM Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        elapsed_gpu += cpu_timer.ElapsedMillis();
    }

    elapsed_gpu /= iterations;
    cpu_timer.Start();

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
	    SizeT num_matches_cpu =  ReferenceSM(
            *graph_query, *graph_data, quiet_mode);
	    if(num_matches_cpu == num_matches_gpu)
	    {
            if (!quiet_mode) 
                DisplaySolution(
                    *graph_query, *graph_data, h_froms, h_tos, num_matches_cpu);
            if (!quiet_mode) { printf("\nCORRECT.\n"); }

        } else {
            if (!quiet_mode)
            {
                printf("INCORRECT.\n");
                printf("CPU Number of matched subgraphs = %lld\n",
                    (long long) num_matches_cpu);
                printf("GPU Number of matched subgraphs = %lld\n",
                    (long long) num_matches_gpu);
            }
        }
    }

    info->ComputeCommonStats(
        enactor -> enactor_stats.GetPointer(), elapsed_gpu, (VertexId*) NULL);

    // clean up if necessary
    if (problem) delete   problem;
    if (enactor) delete   enactor;
    if (h_froms) delete[] h_froms;
    if (h_tos)   delete[] h_tos;
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
}

///////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////
template <
    typename VertexId,  // use int as the vertex identifier
    typename SizeT   ,  // use int as the graph size type
    typename Value   >  // use int as the value type
int main_(CommandLineArgs *args, int graph_args)
{
    CpuTimer cpu_timer, cpu_timer2;
    cpu_timer.Start();

    Csr <VertexId, SizeT, Value> csr_query(false);  // query graph we process on
    Csr <VertexId, SizeT, Value> csr_data(false);  // data graph we process on
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    info->info["undirected"] = true;  // always convert to undirected
    info->info["debug_mode"] = true;  // debug mode
    if (graph_args == 5)  info->info["node_value"] = true;  // require per node label values

    cpu_timer2.Start();
    info->Init("SM", *args, csr_query, csr_data);
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    graphio::RemoveStandaloneNodes<VertexId, SizeT, Value>(
        &csr_data, args -> CheckCmdLineFlag("quite"));

    RunTests<VertexId, SizeT, Value>(info);  // run test

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
int main_Value(CommandLineArgs *args, int graph_args)
{
// disabled to reduce compile time
//    if (args -> CheckCmdLineFlag("64bit-Value"))
//        return main_<VertexId, SizeT, long long>(args, graph_args);
//    else
        return main_<VertexId, SizeT, int      >(args, graph_args);
}

template <
    typename VertexId>
int main_SizeT(CommandLineArgs *args, int graph_args)
{
    //disabled, because atomicSub does not support long long
    //if (args -> CheckCmdLineFlag("64bit-SizeT"))
    //    return main_Value<VertexId, long long>(args, graph_args);
    //else
        return main_Value<VertexId, int      >(args, graph_args);
}

int main_VertexId(CommandLineArgs *args, int graph_args)
{
    // disabled, because oprtr::filter::KernelPolicy::SmemStorage is too large for 64bit VertexId
    //if (args -> CheckCmdLineFlag("64bit-VertexId"))
    //    return main_SizeT<long long>(args, graph_args);
    //else 
        return main_SizeT<int      >(args, graph_args);
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

    return main_VertexId(&args, graph_args);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
