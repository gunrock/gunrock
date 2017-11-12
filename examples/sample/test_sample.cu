// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief Simple test driver program for sample problem.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Sample Problem includes
#include <gunrock/app/sample/sample_enactor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes for CPU Dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sample;

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
        "[--src=<Vertex-ID|randomize|largestdegree>]\n"
        "                          Begins traversal from the source (Default: 0).\n"
        "                          If randomize: from a random source vertex.\n"
        "                          If largestdegree: from largest degree vertex.\n"
        "[--quick]                 Skip the CPU reference validation process.\n"
        "[--mark-pred]             Keep both label info and predecessor info.\n"
        "[--disable-size-check]    Disable frontier queue size check.\n"
        "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
        "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--in-sizing=<in/out_queue_scale_factor>]\n"
        "                          Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--traversal-mode=<0|1>]  Set traversal strategy, 0 for Load-Balanced\n"
        "                          1 for Dynamic-Cooperative (Default: dynamic\n"
        "                          determine based on average degree).\n"
        "[--partition-method=<random|biasrandom|clustered|metis>]\n"
        "                          Choose partitioner (Default use random).\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays the result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution (SizeT num_nodes)
{
    if (num_nodes > 40) num_nodes = 40;

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        //PrintValue(data_to_display[i]);
        printf(" ");
    }
    printf("]\n");
}

/******************************************************************************
 * Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference implementation for sample problem.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
void Reference(
    const std::string &fname,
    Csr<VertexId, SizeT, Value> &graph,
    bool                              quiet)
{
    // Add CPU Implementation here.   
    // Write graph to txt file and generate random edge weights [0,64)
    graph.WriteToLigraFile(
        fname.c_str(),
        graph.nodes,
        graph.edges,
        graph.row_offsets,
        graph.column_indices);
}


/**
 * @brief Run tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value>
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info)
{
    typedef SampleProblem < VertexId,
            SizeT,
            Value > Problem;

    typedef SampleEnactor < Problem > Enactor;

    // parse configurations from mObject info
    Csr<VertexId, SizeT, Value> *graph = info->csr_ptr;
    VertexId    src                 = info->info["source_vertex"    ].get_int64();
    int         max_grid_size       = info->info["max_grid_size"    ].get_int  ();
    int         num_gpus            = info->info["num_gpus"         ].get_int  ();
    double      max_queue_sizing    = info->info["max_queue_sizing" ].get_real ();
    double      max_queue_sizing1   = info->info["max_queue_sizing1"].get_real ();
    double      max_in_sizing       = info->info["max_in_sizing"    ].get_real ();
    std::string partition_method    = info->info["partition_method" ].get_str  ();
    double      partition_factor    = info->info["partition_factor" ].get_real ();
    int         partition_seed      = info->info["partition_seed"   ].get_int  ();
    bool        quiet_mode          = info->info["quiet_mode"       ].get_bool ();
    bool        quick_mode          = info->info["quick_mode"       ].get_bool ();
    bool        stream_from_host    = info->info["stream_from_host" ].get_bool ();
    std::string traversal_mode      = info->info["traversal_mode"   ].get_str  ();
    bool        instrument          = info->info["instrument"       ].get_bool ();
    bool        debug               = info->info["debug_mode"       ].get_bool ();
    bool        size_check          = info->info["size_check"       ].get_bool ();
    int         iterations          = info->info["num_iteration"    ].get_int  ();
    std::string src_type            = info->info["source_type"      ].get_str  (); 
    int      src_seed               = info->info["source_seed"      ].get_int  ();
    int      communicate_latency    = info->info["communicate_latency"].get_int (); 
    float    communicate_multipy    = info->info["communicate_multipy"].get_real();
    int      expand_latency         = info->info["expand_latency"    ].get_int (); 
    int      subqueue_latency       = info->info["subqueue_latency"  ].get_int (); 
    int      fullqueue_latency      = info->info["fullqueue_latency" ].get_int (); 
    int      makeout_latency        = info->info["makeout_latency"   ].get_int (); 
    if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;

    CpuTimer    cpu_timer;
    cudaError_t retval              = cudaSuccess;
    if (max_queue_sizing < 1.2) max_queue_sizing=1.2;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side array (for both reference and GPU-computed results)

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
        if (retval = util::GRError(cudaMemGetInfo(&(org_size[gpu]), &dummy),
            "cudaMemGetInfo failed", __FILE__, __LINE__)) return retval;
    }

    // Allocate problem on GPU
    Problem *problem = new Problem;
    if (retval = util::GRError(problem->Init(
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
        "Sample Problem Init failed", __FILE__, __LINE__))
        return retval;

    // Allocate enactor map
    Enactor* enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    if (retval = util::GRError(enactor->Init(
        context, problem, max_grid_size, traversal_mode),
        "Sample Enactor Init failed", __FILE__, __LINE__))
        return retval;

    enactor -> communicate_latency = communicate_latency;
    enactor -> communicate_multipy = communicate_multipy;
    enactor -> expand_latency      = expand_latency;
    enactor -> subqueue_latency    = subqueue_latency;
    enactor -> fullqueue_latency   = fullqueue_latency;
    enactor -> makeout_latency     = makeout_latency;

    if (retval = util::SetDevice(gpu_idx[0])) return retval;
    if (retval = util::latency::Test(
        streams[0], problem -> data_slices[0] -> latency_data,
        communicate_latency,
        communicate_multipy,
        expand_latency,
        subqueue_latency,
        fullqueue_latency,
        makeout_latency)) return retval;

    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // perform Sample Problem enactor
    double total_elapsed  = 0.0;
    double single_elapsed = 0.0;
    double max_elapsed    = 0.0;
    double min_elapsed    = 1e10;
    json_spirit::mArray process_times;
    if (src_type == "random2")
    {
        if (src_seed == -1) src_seed = time(NULL);
        if (!quiet_mode)
            printf("src_seed = %d\n", src_seed);
        srand(src_seed);
    }
    if (!quiet_mode) printf("Using traversal mode %s\n", traversal_mode.c_str());
    for (int iter = 0; iter < iterations; ++iter)
    {
        if (src_type == "random2")
        {
            bool src_valid = false;
            while (!src_valid)
            {
                src = rand() % graph -> nodes;
                if (graph -> row_offsets[src] != graph -> row_offsets[src+1])
                    src_valid = true;
            }
        }

        if (retval = util::GRError(problem->Reset(
            enactor->GetFrontierType(),
            max_queue_sizing, max_queue_sizing1),
            "SSSP Problem Data Reset Failed", __FILE__, __LINE__))
            return retval;

        if (retval = util::GRError(enactor->Reset(),
            "SSSP Enactor Reset failed", __FILE__, __LINE__))
            return retval;

        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu]))
                return retval;
            if (retval = util::GRError(cudaDeviceSynchronize(),
                "cudaDeviceSynchronize failed", __FILE__, __LINE__))
                return retval;
        }

        if (!quiet_mode)
        {
            printf("__________________________\n"); fflush(stdout);
        }
        cpu_timer.Start();
        if (retval = util::GRError(enactor->Enact(src, traversal_mode),
            "SSSP Problem Enact Failed", __FILE__, __LINE__))
            return retval;
        cpu_timer.Stop();
        single_elapsed = cpu_timer.ElapsedMillis();
        total_elapsed += single_elapsed;
        process_times.push_back(single_elapsed);
        if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
        if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;
        if (!quiet_mode)
        {
            printf("--------------------------\n"
                "iteration %d elapsed: %lf ms, src = %lld, #iteration = %lld\n",
                iter, single_elapsed, (long long)src,
                (long long)enactor -> enactor_stats -> iteration);
            fflush(stdout);
        }
    }
    total_elapsed /= iterations;
    info -> info["process_times"] = process_times;
    info -> info["min_process_time"] = min_elapsed;
    info -> info["max_process_time"] = max_elapsed;

    // compute reference CPU SSSP solution for source-distance
    if (!quick_mode)
    {
        if (!quiet_mode) { printf("Computing reference value ...\n"); }
        Reference<VertexId, SizeT, Value>(
            info->info["dataset"].get_str(),
            *graph,
            quiet_mode);
        if (!quiet_mode) { printf("\n"); }
    }

    cpu_timer.Start();
    // Copy out results
    if (retval = util::GRError(problem->Extract(),
        "Sample Problem Data Extraction Failed", __FILE__, __LINE__))
        return retval;

    if (!quiet_mode)
    {
        // Display Solution
        DisplaySolution<VertexId, SizeT>(graph->nodes);
    }

    // Clean up
    if (org_size        ) {delete[] org_size        ; org_size         = NULL;}
    if (enactor         )
    {
        if (retval = util::GRError(enactor -> Release(),
            "BFS Enactor Release failed", __FILE__, __LINE__))
            return retval;
        delete   enactor         ; enactor          = NULL;
    }
    if (problem         )
    {
        if (retval = util::GRError(problem -> Release(),
            "BFS Problem Release failed", __FILE__, __LINE__))
            return retval;
        delete   problem         ; problem          = NULL;
    }
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    return retval;
}

/******************************************************************************
* Main
******************************************************************************/

template <
    typename VertexId,  // Use int as the vertex identifier
    typename SizeT,     // Use int as the graph size type
    typename Value>     // Use int as the value type
int main_(CommandLineArgs *args)
{
    CpuTimer cpu_timer, cpu_timer2;
    cpu_timer.Start();
    Csr <VertexId, SizeT, Value> csr(false);  // graph we process on
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    info->info["undirected"] = args -> CheckCmdLineFlag("undirected");
    info->info["edge_value"] = true;  // require per edge weight values
    info->info["random_edge_value"] = args -> CheckCmdLineFlag("random-edge-value");

    cpu_timer2.Start();
    info->Init("SSSP", *args, csr);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    cudaError_t retval = RunTests<VertexId, SizeT, Value>(info);  // run test
    cpu_timer.Stop();
    info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!(info->info["quiet_mode"].get_bool()))
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject
    return retval;
}

template <
    typename VertexId, // the vertex identifier type, usually int or long long
    typename SizeT   > // the size tyep, usually int or long long
int main_Value(CommandLineArgs *args)
{
// Disabled becaus atomicMin(long long*, long long) is not available
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
    if (args -> CheckCmdLineFlag("64bit-SizeT"))
        return main_Value<VertexId, long long>(args);
    else
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
