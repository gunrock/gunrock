// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_rw.cu
 *
 * @brief Simple test driver program for random walk problem.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <utility>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <map>
#include <iostream>
#include <random>
#include <map>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/market.cuh>


// RW Problem includes
#include <gunrock/app/rw/rw_enactor.cuh>
#include <gunrock/app/rw/rw_problem.cuh>
#include <gunrock/app/rw/rw_functor.cuh>


// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>


using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::rw;

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
        "[--walk_length]           Set number of walk  (Default: 1).\n"
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
        "[--mode=<num>]            Mode of GPU random walk, 0:raw, 1:device, 2:block\n"
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


template<typename VertexId, 
         typename SizeT>
void DisplaySolution (VertexId *h_paths, SizeT nodes, SizeT walk_length)
{
    
    SizeT limit = nodes > 40 ? 40 : nodes;
    SizeT walkLimit = walk_length > 11 ? 10 : walk_length; 
    printf("==> random walk output paths:\n");

    printf("[");
    for (SizeT i = 0; i < limit; ++i)
    {   
        //printf("%lld (%lld): ", (long long)h_paths[i], (long long)h_neighbor[i]);
        printf("%lld : ", (long long)h_paths[i]);
        for(SizeT j = 1; j < walkLimit; ++j){
            printf("%lld ", (long long)h_paths[j*nodes+i]);
        }
        printf("\n");
    }
    printf("]\n");
}

/**
 * @brief Displays the result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] num_nodes Number of nodes in the graph.
 */


template<typename VertexId, 
         typename SizeT>
void StoreSolution (VertexId *h_paths, SizeT nodes, SizeT walk_length, int mode)
{
    
    SizeT limit = nodes > 40 ? 40 : nodes;
    SizeT walkLimit = walk_length > 11 ? 10 : walk_length; 
    printf("==> random walk output paths:\n");

    printf("[");
    for (SizeT i = 0; i < limit; ++i)
    {   
        //printf("%lld (%lld): ", (long long)h_paths[i], (long long)h_neighbor[i]);
        printf("%lld : ", (long long)h_paths[i]);
        for(SizeT j = 1; j < walkLimit; ++j){
            printf("%lld ", (long long)h_paths[j*nodes+i]);
        }
        printf("\n");
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
void ReferenceRW(
    Csr<VertexId, SizeT, Value> &graph,
    int                         walk_length,
    bool                        quiet = false)
{ 
    //assume the input data are sorted 
    int len = walk_length; 
    std::map<SizeT, std::vector<SizeT>> arr;
    for (SizeT i = 0; i < graph.nodes; ++i)
    {
     
        if(graph.row_offsets[i] != graph.row_offsets[i+1]){
		std::vector<SizeT> l;
		for (SizeT j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
        	{ 
			l.push_back(graph.column_indices[j]);
        	}

		arr[i] = l;
	}
    }
  
    std::vector<SizeT> curr;
    std::random_device rand;
    std::mt19937 engine{rand()}; 
    std::map<SizeT, std::vector<SizeT>> output;    
   
    CpuTimer cpu_timer;
    cpu_timer.Start();
    for(VertexId i = 0; i < graph.nodes; ++i){
	if(arr.count(i) == 1){
        	curr = arr[i];
        	std::vector<SizeT> out;
        	out.push_back(i);
        	for(VertexId j = 1; j < len; ++j){
            		std::uniform_int_distribution<int> dist(0, curr.size()-1);
            		VertexId next = curr[dist(engine)];
            		out.push_back(next);
            		curr = arr[next];
        	}	
        	output[i] = out;
	}
    }
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    
    if(!quiet){
        printf("Walklength: %d.\n CPU RW computation complete in %lf msec.\n",
                   len, elapsed);
    }
    /*debug purpose
    printf("example out(walk length limit to 5): \n");
    for(int i = 0; i < graph.nodes; i++){
        for(int k = 0; k < length; k++){
            printf("%d ", output[i][k]);
        }
        printf("\n");
    }
    */
  
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
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info, int mode)
{
    typedef RWProblem < VertexId,
            SizeT,
            Value > Problem;

    typedef RWEnactor < Problem > Enactor;

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
    //use this flag for sorted enactor
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
    int      walk_length            = info->info["walk_length"       ].get_int ();
    std::string output_filename     = info->info["output_filename"   ].get_str (); 

    
    //if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;


    CpuTimer    cpu_timer;
    cudaError_t retval              = cudaSuccess;
    if (max_queue_sizing < 0) max_queue_sizing=1.0;
    if(max_in_sizing < 0) max_in_sizing = 1.0;
    iterations = 1;
    info -> info["search_depth"] = iterations;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side array (for both reference and GPU-computed results)
    /*
    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
        if (retval = util::GRError(cudaMemGetInfo(&(org_size[gpu]), &dummy),
            "cudaMemGetInfo failed", __FILE__, __LINE__)) return retval;
    }
    */
    SizeT nodes = graph->nodes;
    VertexId *h_paths;
    VertexId *h_trailing_paths;

    if(mode == BLOCK){
        int grid = nodes/(THREAD_BLOCK*ELEMS_PER_THREAD);
        int block_data = ELEMS_PER_THREAD*THREAD_BLOCK*grid;
        int trailing = nodes - block_data;

        h_paths = (VertexId*)malloc(sizeof(VertexId) * block_data * walk_length);
        h_trailing_paths = (VertexId*)malloc(sizeof(VertexId) * trailing * walk_length);
    }else{
        h_paths = (VertexId*)malloc(sizeof(VertexId) * nodes * walk_length);

    }

    
 
    if (!quick_mode){
    	ReferenceRW(*graph, walk_length, quiet_mode);
    }

    // Allocate problem on GPU
    Problem *problem = new Problem(walk_length, mode);
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
        "RW Problem Init failed", __FILE__, __LINE__))
        return retval;

    // Allocate enactor map
    Enactor* enactor = new Enactor(num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor->Init(
        context, problem, max_grid_size),
        "Enactor Init failed", __FILE__, __LINE__);

    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    double total_elapsed  = 0.0;
    double single_elapsed = 0.0;
    double max_elapsed    = 0.0;
    double min_elapsed    = 1e10;
    json_spirit::mArray process_times;
    if (retval = util::GRError(problem->Reset(enactor->GetFrontierType(),
                                        max_queue_sizing, max_queue_sizing1),
                                        "RW Problem Data Reset Failed", __FILE__, __LINE__))
        return retval;
    
    if (retval = util::GRError(enactor->Reset(),
            "RW Enactor Reset failed", __FILE__, __LINE__))
        return retval;


    cpu_timer.Start();
    if (retval = util::GRError(enactor->Enact(mode),
            "RW Problem Enact Failed", __FILE__, __LINE__))
        return retval;
    cpu_timer.Stop();

    single_elapsed += cpu_timer.ElapsedMillis();
    total_elapsed += single_elapsed;
    process_times.push_back(single_elapsed);
        if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
        if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;

    if (!quiet_mode)
    {
        printf("--------------------------\n"
                "iteration %d elapsed: %lf ms, #iteration = %lld\n",
                iterations, single_elapsed,
                (long long)enactor -> enactor_stats -> iteration);
            fflush(stdout);
	// printf("GPU RW computation completed in %lf msec.\n", single_elapsed);
    }

    total_elapsed /= iterations;
    info -> info["process_times"] = process_times;
    info -> info["min_process_time"] = min_elapsed;
    info -> info["max_process_time"] = max_elapsed;
    info -> info["elapsed"] = single_elapsed;
    /*
    if (!quick_mode)
    {
        if (!quiet_mode) { printf("Computing reference value ...\n"); }
        Reference<VertexId, SizeT, Value>(
            info->info["dataset"].get_str(),
            *graph,
            quiet_mode);
        if (!quiet_mode) { printf("\n"); }
    }*/


    // Copy out GPU results
    cpu_timer.Start();
    if (retval = util::GRError(problem->Extract(h_paths),
        "RW Problem Data Extraction Failed", __FILE__, __LINE__))
        return retval;


    if (!quiet_mode)
    {
        // Display Solution
        DisplaySolution(h_paths,
                        nodes,
                        walk_length);
    }

    if(!output_filename.empty()){
        StoreSolution(h_paths,
                  nodes,
                  walk_length,
                  mode);
    }

    // Clean up
    cpu_timer.Start();

    if (enactor         )
    {
        if (retval = util::GRError(enactor -> Release(),
            "RW Enactor Release failed", __FILE__, __LINE__))
            return retval;
        delete   enactor         ; enactor          = NULL;
    }
    
    if (problem         )
    {
        if (retval = util::GRError(problem -> Release(),
            "RW Problem Release failed", __FILE__, __LINE__))
            return retval;

        delete   problem         ; 
        problem          = NULL;
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

    /** additional arg parameter for rw testing**/
    //bool block_sort = args -> CheckCmdLineFlag("block_sort");
    //bool device_sort = args -> CheckCmdLineFlag("device_sort");
    int mode;

    if (args->CheckCmdLineFlag("mode"))
        {
            args->GetCmdLineArgument("mode", mode);
        }
    //printf("mode %d\n", mode);



    info->info["edge_value"] = false;  // require per edge weight values
    cpu_timer2.Start();
    info->Init("RW", *args, csr);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    cudaError_t retval = RunTests<VertexId, SizeT, Value>(info, mode);  // run test

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
    typename VertexId, //the vertex identifier type, usually int or long long
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
    /*
    if (args -> CheckCmdLineFlag("64bit-SizeT"))
        return main_Value<VertexId, long long>(args);
    else*/
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
