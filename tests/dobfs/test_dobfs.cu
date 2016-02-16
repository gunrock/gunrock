// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// DOBFS includes
#include <gunrock/app/dobfs/dobfs_enactor.cuh>
#include <gunrock/app/dobfs/dobfs_problem.cuh>
#include <gunrock/app/dobfs/dobfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;
using namespace gunrock::app::dobfs;


/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf (
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
        "[--idempotence]           Whether or not to enable idempotent operation.\n"
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
        "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
        "                          (graph-edges * <factor>). (Default: 1.0)\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--traversal-mode=<0|1>]  Set traversal strategy, 0 for Load-Balanced\n"
        "                          1 for Dynamic-Cooperative (Default: dynamic\n"
        "                          determine based on average degree).\n"
        "[--alpha=<alpha>]         Alpha factor for direction-optimizing.\n"
        "                          Switching to backward approach.\n"
        "[--beta=<beta>]           Beta factor for direction-optimizing.\n"
        "                          Switching back to the forward approach.\n" 
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] preds Predecessor node id for each node.
 * @param[in] nodes Number of nodes in the graph.
 * @param[in] MARK_PREDECESSORS Whether to show predecessor of each node.
 * @param[in] ENABLE_IDEMPOTENCE Whether to enable idempotence mode.
 */
template <
    typename VertexId, 
    typename SizeT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE>
void DisplaySolution(
    VertexId *source_path,
    VertexId *preds,
    SizeT nodes)
{
    if (nodes > 40) nodes = 40;
    printf("\nFirst %lld labels of the GPU result.\n", (long long)nodes);

    printf("[");
    for (VertexId i = 0; i < nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE)
        {
            printf(",");
            PrintValue(preds[i]);
        }
        printf(" ");
    }
    printf("]\n");
}

/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference BFS ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source_path Host-side vector to store CPU computed labels for each node
 * @param[in] predecessor Host-side vector to store CPU computed predecessor for each node
 * @param[in] src Source node where BFS starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void ReferenceBFS(
    const Csr<VertexId, SizeT, Value> *graph,
    VertexId                          *source_path,
    VertexId                          *predecessor,
    VertexId                          src,
    bool                              quiet = false)
{
    // Initialize labels
    for (VertexId i = 0; i < graph->nodes; ++i)
    {
        source_path[i] = ENABLE_IDEMPOTENCE ? -1 : util::MaxValue<VertexId>() - 1;
        if (MARK_PREDECESSORS)
        {
            predecessor[i] = -1;
        }
    }
    source_path[src] = 0;
    VertexId search_depth = 0;

    // Initialize queue for managing previously-discovered nodes
    std::deque<VertexId> frontier;
    frontier.push_back(src);

    // Perform BFS
    CpuTimer cpu_timer;
    cpu_timer.Start();
    while (!frontier.empty())
    {
        // Dequeue node from frontier
        VertexId dequeued_node = frontier.front();
        frontier.pop_front();
        VertexId neighbor_dist = source_path[dequeued_node] + 1;

        // Locate adjacency list
        SizeT edges_begin = graph->row_offsets[dequeued_node];
        SizeT edges_end = graph->row_offsets[dequeued_node + 1];

        for (SizeT edge = edges_begin; edge < edges_end; ++edge)
        {
            //Lookup neighbor and enqueue if undiscovered
            VertexId neighbor = graph->column_indices[edge];
            if (source_path[neighbor] > neighbor_dist || source_path[neighbor] == -1)
            {
                source_path[neighbor] = neighbor_dist;
                if (MARK_PREDECESSORS)
                {
                    predecessor[neighbor] = dequeued_node;
                }
                if (search_depth < neighbor_dist)
                {
                    search_depth = neighbor_dist;
                }
                frontier.push_back(neighbor);
            }
        }
    }

    if (MARK_PREDECESSORS)
    {
        predecessor[src] = -1;
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    if (!quiet)
    {
        printf("CPU BFS finished in %lf msec. cpu_search_depth: %d\n",
               elapsed, search_depth);
    }
}

/**
 * @brief Run BFS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void RunTests(Info<VertexId, SizeT, Value> *info)
{
    typedef DOBFSProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS,
            ENABLE_IDEMPOTENCE>
            //(MARK_PREDECESSORS && ENABLE_IDEMPOTENCE) >
            Problem; // does not use double buffer
    typedef DOBFSEnactor <Problem>
            Enactor;

    Csr<VertexId, SizeT, Value> *csr = info->csr_ptr;
    Csr<VertexId, SizeT, Value> *csc = info->csc_ptr;
    VertexId src                    = info->info["source_vertex"    ].get_int64();
    int      max_grid_size          = info->info["max_grid_size"    ].get_int  ();
    int      num_gpus               = info->info["num_gpus"         ].get_int  ();
    double   max_queue_sizing       = info->info["max_queue_sizing" ].get_real ();
    double   max_queue_sizing1      = info->info["max_queue_sizing1"].get_real ();
    double   max_in_sizing          = info->info["max_in_sizing"    ].get_real ();
    std::string partition_method    = info->info["partition_method" ].get_str  ();
    double   partition_factor       = info->info["partition_factor" ].get_real ();
    int      partition_seed         = info->info["partition_seed"   ].get_int  ();
    bool     quiet_mode             = info->info["quiet_mode"       ].get_bool ();
    bool     quick_mode             = info->info["quick_mode"       ].get_bool ();
    bool     undirected             = info->info["undirected"       ].get_bool ();
    bool     stream_from_host       = info->info["stream_from_host" ].get_bool ();
    bool     instrument             = info->info["instrument"       ].get_bool (); 
    bool     debug                  = info->info["debug_mode"       ].get_bool (); 
    bool     size_check             = info->info["size_check"       ].get_bool (); 
    int      iterations             = 1; // force to 1 info->info["num_iteration"].get_int();
    double   alpha                  = info->info["alpha"            ].get_real ();
    double   beta                   = info->info["beta"             ].get_real ();
    CpuTimer cpu_timer;

    cpu_timer.Start();
    if (!quiet_mode)
    {
        printf(" alpha = %.4f, beta = %.4f\n", alpha, beta);
    }

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) 
        gpu_idx[i] = device_list[i].get_int();
    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {   
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }  

    // TODO: add memory size check and remove this
    if (max_queue_sizing < 1.0 * csr->edges / csr->nodes)
    {
        max_queue_sizing = 1.0 * csr->edges / csr->nodes;
        max_queue_sizing1 = max_queue_sizing;
    }

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*  )info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // allocate host-side label array (for both reference and GPU results)
    VertexId *ref_labels       = (quick_mode) ? NULL : new VertexId [csr -> nodes];
    VertexId *h_labels         = new VertexId[csr -> nodes];
    VertexId *h_preds          = MARK_PREDECESSORS ? new VertexId[csr -> nodes] : NULL;
    VertexId *ref_preds        = (!quick_mode && MARK_PREDECESSORS) ? new VertexId[csr -> nodes] : NULL;

    // Allocate problem on GPU
    Problem *problem = new Problem((MARK_PREDECESSORS && ENABLE_IDEMPOTENCE));
    util::GRError(problem->Init(
        stream_from_host,
        *csr,
        *csc,
        undirected,
        alpha,
        beta,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "Problem DOBFS Initialization Failed", __FILE__, __LINE__);

    // Allocate BFS enactor map
    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    util::GRError(enactor -> Init(
        context, problem, max_grid_size),
        "DOBFS Enactor Init failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // compute reference CPU BFS solution
    if (!quick_mode)
    {
        if (!quiet_mode)
        {
            printf("Computing reference value ...\n");
        }
        ReferenceBFS<VertexId, SizeT, Value,
            MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>(
            csr,
            ref_labels,
            ref_preds,
            src,
            quiet_mode);
        if (!quiet_mode)
        {
            printf("\n");
        }
    }

    double elapsed = 0.0f;
    // Perform BFS

    for (int iter = 0; iter < iterations; ++iter)
    {
        util::GRError(problem->Reset(
            src, enactor -> GetFrontierType(), 
            max_queue_sizing, max_queue_sizing1),
            "DOBFS Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(enactor -> Reset());

        cpu_timer.Start();
        util::GRError(enactor -> Enact(src),
            "DOBFS Problem Enact Failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        elapsed += cpu_timer.ElapsedMillis();
    }
    elapsed /= iterations;

    cpu_timer.Start();
    // Copy out results
    util::GRError(problem->Extract(h_labels, h_preds),
        "DOBFS Problem Data Extraction Failed", __FILE__, __LINE__);

    // Verify the result
    if (!quick_mode && !quiet_mode)
    {
        printf("Label Validity: ");
        int num_errors = CompareResults(
            h_labels, ref_labels,
            csr ->nodes, true, quiet_mode);
        if (num_errors > 0)
        {   
            printf("%d errors occurred.", num_errors);
        }   
        printf("\n");

        if (MARK_PREDECESSORS)
        {   
            printf("Predecessor Validity: \n");
            num_errors = 0;
            for (VertexId v=0; v< csr->nodes; v++)
            {   
                if (h_labels[v] ==  
                    (ENABLE_IDEMPOTENCE ? -1 : util::MaxValue<VertexId>() - 1)) 
                    continue; // unvisited vertex
                if (v == src && h_preds[v] == -1) continue; // source vertex
                VertexId pred = h_preds[v];
                if (pred >= csr->nodes || pred < 0)
                {   
                    //if (num_errors == 0)
                        printf("INCORRECT: pred[%d] : %d out of bound\n", v, pred);
                    num_errors ++; 
                    continue;
                }   
                if (h_labels[v] != h_labels[pred] + 1)
                {   
                    //if (num_errors == 0)
                        printf("INCORRECT: label[%d] (%d) != label[%d] (%d) + 1\n", 
                            v, h_labels[v], pred, h_labels[pred]);
                    num_errors ++; 
                    continue;
                }   
    
                bool v_found = false;
                for (SizeT t = csr->row_offsets[pred]; t < csr->row_offsets[pred+1]; t++)
                if (v == csr->column_indices[t])
                {   
                    v_found = true;
                    break;
                }   
                if (!v_found)
                {   
                    //if (num_errors == 0)
                        printf("INCORRECT: Vertex %d not in Vertex %d's neighbor list\n",
                            v, pred);
                    num_errors ++; 
                    continue;
                }   
            }   

            if (num_errors > 0)
            {   
                printf("%d errors occurred.", num_errors);
            } else printf("CORRECT");
            printf("\n");
        }   
    }

    if (!quiet_mode)
    {
        DisplaySolution<VertexId, SizeT, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>
            (h_labels, h_preds, csr->nodes);
    }

    info->ComputeTraversalStats(  // compute running statistics
        enactor -> enactor_stats.GetPointer(), elapsed, h_labels);

    if (!quiet_mode)
    {
        printf("\n\tMemory Usage(B)\t");
        for (int gpu = 0; gpu < num_gpus; gpu++)
            if (num_gpus > 1)
            {
                if (gpu != 0)
                {
                    printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1",
                           gpu, gpu, gpu, gpu);
                }
                else
                {
                    printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
                }
            }
            else
            {
                printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
            }
        if (num_gpus > 1)
        {
            printf(" #keys%d", num_gpus);
        }
        printf("\n");
        double max_queue_sizing_[2] = {0, 0 }, max_in_sizing_ = 0;
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {   
            size_t gpu_free, dummy;
            cudaSetDevice(gpu_idx[gpu]);
            cudaMemGetInfo(&gpu_free, &dummy);
            printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
            for (int i = 0; i < num_gpus; i++)
            {   
                for (int j = 0; j < 2; j++)
                {   
                    SizeT x = problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
                    printf("\t %lld", (long long) x); 
                    double factor = 1.0 * x / (num_gpus > 1 ? problem->graph_slices[gpu]->in_counter[i] : problem->graph_slices[gpu]->nodes);
                    if (factor > max_queue_sizing_[j])
                    {   
                        max_queue_sizing_[j] = factor;
                    }   
                }   
                if (num_gpus > 1 && i != 0 ) 
                {   
                    for (int t = 0; t < 2; t++)
                    {   
                        SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                        printf("\t %lld", (long long) x); 
                        double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
                        if (factor > max_in_sizing_)
                        {
                            max_in_sizing_ = factor;
                        }
                    }
                }
            }
            if (num_gpus > 1)
            {
                printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
            }
            printf("\n");
        }
        printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
        if (num_gpus > 1)
        {
            printf("\t in_sizing =\t %lf", max_in_sizing_);
        }
        printf("\n");
    }

    // Clean up
    if (problem   ) {delete   problem   ; problem    = NULL;}
    if (enactor   ) {delete   enactor   ; enactor    = NULL;}
    if (ref_preds ) {delete[] ref_preds ; ref_preds  = NULL;}
    if (ref_labels) {delete[] ref_labels; ref_labels = NULL;}
    if (h_labels  ) {delete[] h_labels  ; h_labels   = NULL;}
    if (h_preds   ) {delete[] h_preds   ; h_preds    = NULL;}
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();

    //cudaDeviceSynchronize();
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
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    //bool        INSTRUMENT,
    //bool        DEBUG,
    //bool        SIZE_CHECK,
    bool        MARK_PREDECESSORS >
void RunTests_enable_idempotence(Info<VertexId, SizeT, Value> *info)
{
    if (info->info["idempotent"].get_bool())
        RunTests <VertexId, SizeT, Value, MARK_PREDECESSORS, true > (info);
    else
        RunTests <VertexId, SizeT, Value, MARK_PREDECESSORS, false> (info);
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
    typename    VertexId,
    typename    SizeT,
    typename    Value>
    //bool        INSTRUMENT,
    //bool        DEBUG,
    //bool        SIZE_CHECK >
void RunTests_mark_predecessors(Info<VertexId, SizeT, Value> *info)
{
    if (info->info["mark_predecessors"].get_bool())
        RunTests_enable_idempotence<VertexId, SizeT, Value, true> (info);
    else
        RunTests_enable_idempotence<VertexId, SizeT, Value, false> (info);
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

    Csr <VertexId, SizeT, Value> csr(false);  // CSR graph we process on
    Csr <VertexId, SizeT, Value> csc(false);  // CSC graph we process on
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    info->info["undirected"] = args -> CheckCmdLineFlag("undirected");

    cpu_timer2.Start();
    info->Init("DOBFS", *args, csr, csc);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    RunTests_mark_predecessors<VertexId, SizeT, Value>(info);  // run test

    cudaDeviceReset();
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
// disabled becaused of filter smem oversize issue
//    if (args -> CheckCmdLineFlag("64bit-VertexId"))
//        return main_SizeT<long long>(args);
//    else 
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
