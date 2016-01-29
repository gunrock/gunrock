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
#include <gunrock/app/sm/sm_enactor.cuh>
#include <gunrock/app/sm/sm_problem.cuh>
#include <gunrock/app/sm/sm_functor.cuh>

// gunrock abstraction graph operators
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sm;

// ----------------------------------------------------------------------------
// Housekeeping Routines
// ----------------------------------------------------------------------------
void Usage()
{
    printf(
        "test <graph-type> <query graph file> <data graph file> [graph-type-arguments]\n"
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
void DisplaySolution(const Csr<VertexId, Value, SizeT> &graph_query, 
		     const Csr<VertexId, Value, SizeT> &graph_data,
		     VertexId *h_froms,
		     VertexId *h_tos,
		     unsigned int num_matches)
{
    // TODO(developer): code to print out results
    printf("Number of matched subgraphs: %u.\n",num_matches);

}

// ----------------------------------------------------------------------------
// Testing Routines
// ----------------------------------------------------------------------------

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
void SMReference(const Csr<VertexId, Value, SizeT> &graph_query,
		     const Csr<VertexId, Value, SizeT> &graph_data,
		     VertexId *h_froms,
		     VertexId *h_tos) 
{
    // initialization

    // perform subgraph matching calculation

    CpuTimer cpu_timer;
    cpu_timer.Start();

    // TODO(developer): CPU validation code here

    cpu_timer.Stop();

    float cpu_elapsed = cpu_timer.ElapsedMillis();
    printf("CPU reference finished in %lf ms.\n\n", cpu_elapsed);
}

/**
 * @brief Subgraph Matching test entry
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
    bool DEBUG,
    bool SIZE_CHECK >
void RunTest(Info<VertexId, Value, SizeT> *info)
{
    typedef SMProblem < VertexId, SizeT, Value,
            true,   // MARK_PREDECESSORS
            false,  // ENABLE_IDEMPOTENCE
            true >  // USE_DOUBLE_BUFFER
	    Problem;

    Csr<VertexId, Value, SizeT>* csr_query = (Csr<VertexId, Value, SizeT>*)info->csr_query_ptr;
    Csr<VertexId, Value, SizeT>* csr_data = (Csr<VertexId, Value, SizeT>*)info->csr_data_ptr;

    std::string partition_method    = info->info["partition_method"].get_str();
    int         max_grid_size       = info->info["max_grid_size"].get_int();
    int         num_gpus            = info->info["num_gpus"].get_int();
    int         iterations          = 1; //force to 1 for now.
    bool        quick_mode          = info->info["quick_mode"].get_bool();
    bool        quiet_mode          = info->info["quiet_mode"].get_bool();
    bool        stream_from_host    = info->info["stream_from_host"].get_bool();
    double      max_queue_sizing    = info->info["max_queue_sizing"].get_real();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr* context             = (ContextPtr*)info->context;
    cudaStream_t *streams = (cudaStream_t*) info->streams;

    //bool     *h_c_set = new bool[csr_data->nodes * csr_query->nodes];
    VertexId *h_froms = (VertexId*)malloc(sizeof(VertexId) * csr_data->edges);
    VertexId *h_tos = (VertexId*)malloc(sizeof(VertexId) * csr_data->edges);
    unsigned int ref_num_matches = 0;

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }
    if(!quiet_mode) printf("\nSUBGRAPH MATCHING TEST\n");

    SMEnactor <
    Problem,
    false,  // INSTRUMENT
    DEBUG,  // DEBUG
    SIZE_CHECK >  // SIZE_CHECK
    enactor(gpu_idx);  // allocate primitive enactor map

    Problem *problem = new Problem;  // allocate primitive problem on GPU


    // copy data from cpu to gpu initialize data members in DataSlice
    util::GRError(
        problem->Init(stream_from_host, *csr_query, *csr_data, num_gpus),
        "SM Problem Initialization Failed", __FILE__, __LINE__);
	

    // TODO: compute reference CPU SM

    //
    // perform SM calculations
    //

    GpuTimer gpu_timer;// record the kernel running time

    float elapsed = 0.0f;
    for (int iter = 0; iter < iterations; ++iter)
    {
	// reset values in DataSlice
        util::GRError(
            problem->Reset(enactor.GetFrontierType(),
                           max_queue_sizing),
            		"SM Problem Data Reset Failed", __FILE__, __LINE__);

        gpu_timer.Start();
	// launch SM enactor
        util::GRError(
            enactor.template Enact<Problem>(*context, problem, max_grid_size),
            "SM Problem Enact Failed", __FILE__, __LINE__);

        gpu_timer.Stop();

        elapsed += gpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;

    if (!quiet_mode)
    {
        printf("GPU - Computation Complete in %lf msec.\n", elapsed);
    }

    // extract results from GPU to CPU
    util::GRError(
        problem->Extract(h_froms,h_tos),
        "SM Problem Data Extraction Failed", __FILE__, __LINE__);

    // TODO: validity




    // TODO: compute reference CPU validation solution
    if (!quick_mode)
    {
        if (!quiet_mode) printf("==> computing reference value ... (currently missing)\n");
        SMReference<VertexId, SizeT, Value>(csr_query, csr_data, h_froms, h_tos);
        if (!quiet_mode) {printf("==> validation: (currently missing)\n");
/*
    		if (reference_check != NULL)
   		{
      		  if (ref_num_matches == problem->num_matches)
      		  {
            		if (!quiet_mode)
               			 printf("CORRECT. Matched Subgraph Count: %d\n", ref_num_matches);
       		  }
       	 	  else
      	   	  {
            		if (!quiet_mode)
           		{
                		printf(
                    		"INCORRECT. Ref Match Count: %d, "
		                "GPU Subgraph Matching Count: %d\n",
                		ref_num_matches, problem->num_matches);
           		}
       		  }
    		}
    		else
   		{
		      if (!quiet_mode)
           		 printf("Matched Subgraph  Count: %lld\n", (long long) problem->num_matches);
   		} */
	}
   }

    if (!quiet_mode) DisplaySolution<VertexId, SizeT, Value>(csr_query, csr_data, h_froms, h_tos, problem->num_matches);  // display solution

    info->ComputeCommonStats(enactor.enactor_stats.GetPointer(), elapsed);

    if (!quiet_mode) info->DisplayStats(false); // display collected statistics

    info->CollectInfo();

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

    // clean up if necessarry
    if (org_size) {delete[] org_size; org_size=NULL;}
    if (problem)  { delete problem; problem = NULL;}
    if (h_froms)     { delete[] h_froms; h_froms = NULL; }
    if (h_tos)     { delete[] h_tos; h_tos = NULL; }

}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam DEBUG
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          DEBUG >
void RunTests_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
        RunTest <VertexId, Value, SizeT, DEBUG,  true>(info);
    else
        RunTest <VertexId, Value, SizeT, DEBUG, false>(info);
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
    typename    VertexId,
    typename    Value,
    typename    SizeT >
void RunTests_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
        RunTests_size_check <VertexId, Value, SizeT,  true>(info);
    else
        RunTests_size_check <VertexId, Value, SizeT, false>(info);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{

    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    printf("number of graphs: %d\n", graph_args);
    if (argc < 3 || graph_args < 2 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    typedef int VertexId;  // use int as the vertex identifier
    typedef int Value;   // use float as the value type
    typedef int SizeT;     // use int as the graph size type

    Csr<VertexId, Value, SizeT> csr_query(false);  // query graph we process on
    Csr<VertexId, Value, SizeT> csr_data(false);  // data graph we process on
    Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info->info["undirected"] = true; // always converted to undirected
    info->info["multi_graph"] = true; // always have two input graphs
    if(graph_args==5) info->info["node_value"] = true; // node have label

    info->Init("SM", args, csr_query, csr_data);  // initialize Info_query structure
    RunTests_debug<VertexId, Value, SizeT>(info);  // run test
    
    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
