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
#include <algorithm>
#include <iostream>
#include <fstream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/track_utils.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <gunrock/util/shared_utils.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

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
        "[--idempotence]           Whether or not to enable idempotent operation.\n"
        "[--instrumented]          Keep kernels statics [Default: Disable].\n"
        "                          total_queued, search_depth and barrier duty.\n"
        "                          (a relative indicator of load imbalance.)\n"
        "[--src=<Vertex-ID|largestdegree|randomize|randomize2|list>]\n"
        "                          Begins traversal from the source (Default: 0).\n"
        "                          If largestdegree: from largest degree vertex.\n"
        "                          If randomize: from a random source vertex.\n"
        "                          If randomize2: from a different random source vertex for each iteration.\n"
        "                          If list: need to provide a source list through --source_list=n0,n1,...,nk\n"
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
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] labels    Search depth from the source for each node.
 * @param[in] preds     Predecessor node id for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 * @param[in] quiet     Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename SizeT,
    bool MARK_PREDECESSORS,
    bool ENABLE_IDEMPOTENCE >
void DisplaySolution(
    VertexId *labels,
    VertexId *preds,
    SizeT     num_nodes,
    bool quiet = false)
{
    if (quiet) { return; }
    // careful: if later code in this
    // function changes something, this
    // return is the wrong thing to do

    if (num_nodes > 40) { num_nodes = 40; }

    printf("\nFirst %lld labels of the GPU result:\n",
        (long long)num_nodes);

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(labels[i]);
        if (MARK_PREDECESSORS) //&& !ENABLE_IDEMPOTENCE)
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
        source_path[i] = /*ENABLE_IDEMPOTENCE ? -1 :*/ util::MaxValue<VertexId>();
        if (MARK_PREDECESSORS)
        {
            predecessor[i] = util::InvalidValue<VertexId>();
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
            if (source_path[neighbor] > neighbor_dist) //|| source_path[neighbor] == -1)
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
        predecessor[src] = util::InvalidValue<VertexId>();
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    search_depth++;

    if (!quiet)
    {
        printf("CPU BFS finished in %lf msec. cpu_search_depth: %lld\n",
               elapsed, (long long)search_depth);
    }
}

/**
 * @brief Run BFS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        MARK_PREDECESSORS,
    bool        ENABLE_IDEMPOTENCE >
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info)
{
    typedef BFSProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS,
            ENABLE_IDEMPOTENCE>
            //(MARK_PREDECESSORS && ENABLE_IDEMPOTENCE) >
            Problem;  // does not use double buffer

    typedef BFSEnactor < Problem>
            //INSTRUMENT,
            //DEBUG,
            //SIZE_CHECK >
            Enactor;

    // parse configurations from mObject info
    Csr<VertexId, SizeT, Value> *graph = info->csr_ptr;
    Csr<VertexId, SizeT, Value> *inv_graph = info->csc_ptr;
    VertexId src                   = info->info["source_vertex"     ].get_int64();
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
    std::string traversal_mode     = info->info["traversal_mode"    ].get_str  ();
    bool     instrument            = info->info["instrument"        ].get_bool ();
    bool     debug                 = info->info["debug_mode"        ].get_bool ();
    bool     size_check            = info->info["size_check"        ].get_bool ();
    int      iterations            = info->info["num_iteration"     ].get_int  ();
    std::string src_type           = info->info["source_type"       ].get_str  ();
    int      src_seed              = info->info["source_seed"       ].get_int  ();
    int      communicate_latency   = info->info["communicate_latency"].get_int ();
    float    communicate_multipy   = info->info["communicate_multipy"].get_real();
    int      expand_latency        = info->info["expand_latency"    ].get_int ();
    int      subqueue_latency      = info->info["subqueue_latency"  ].get_int ();
    int      fullqueue_latency     = info->info["fullqueue_latency" ].get_int ();
    int      makeout_latency       = info->info["makeout_latency"   ].get_int ();
    bool     direction_optimized   = info->info["direction_optimized"].get_bool();
    float    do_a                  = info->info["do_a"              ].get_real();
    float    do_b                  = info->info["do_b"              ].get_real();
    bool     undirected            = info->info["undirected"        ].get_bool();
    if (max_queue_sizing < 0) max_queue_sizing = 6.5;
    if (max_in_sizing < 0) max_in_sizing = 4;
    if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;

    CpuTimer cpu_timer;
    cudaError_t retval             = cudaSuccess;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // allocate host-side label array (for both reference and GPU results)
    VertexId *reference_labels      = new VertexId[graph->nodes];
    VertexId *reference_preds       = new VertexId[graph->nodes];
    VertexId *h_labels              = new VertexId[graph->nodes];
    VertexId *reference_check_label = (quick_mode) ? NULL : reference_labels;
    VertexId *reference_check_preds = NULL;
    VertexId *h_preds               = NULL;

    if (MARK_PREDECESSORS)
    {
        h_preds = new VertexId[graph->nodes];
        if (!quick_mode)
        {
            reference_check_preds = reference_preds;
        }
    }

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
        if (retval = util::GRError( cudaMemGetInfo(&(org_size[gpu]), &dummy),
            "cudaMemGetInfo failed", __FILE__, __LINE__)) return retval;
    }

    Problem* problem = new Problem(direction_optimized, undirected);  // allocate problem on GPU
    if (retval = util::GRError(problem->Init(
        stream_from_host,
        graph,
        inv_graph,
        num_gpus,
        gpu_idx,
        partition_method,
        streams,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "BFS Problem Init failed", __FILE__, __LINE__)) return retval;

    Enactor* enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check, direction_optimized);  // enactor map
    if (retval = util::GRError(enactor->Init(
        context, problem, max_grid_size, traversal_mode),
        "BFS Enactor Init failed", __FILE__, __LINE__))
        return retval;

    enactor -> communicate_latency = communicate_latency;
    enactor -> communicate_multipy = communicate_multipy;
    enactor -> expand_latency      = expand_latency;
    enactor -> subqueue_latency    = subqueue_latency;
    enactor -> fullqueue_latency   = fullqueue_latency;
    enactor -> makeout_latency     = makeout_latency;
    enactor -> do_a                = do_a;
    enactor -> do_b                = do_b;

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

    // perform BFS
    double total_elapsed = 0.0;
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
    if (!quiet_mode)
        printf("Using traversal-mode %s\n", traversal_mode.c_str());

    json_spirit::mArray source_list;
    if (src_type == "list")
        source_list = info->info["source_list"].get_array();
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
        } else if (src_type == "list")
        {
            if (source_list.size() == 0)
            {
                if (!quiet_mode)
                    printf("No source list found. Use 0 as source.\n");
                src = 0;
            } else {
                src = source_list[iter].get_int();
            }
        }

        if (retval = util::GRError(problem->Reset(
            src, enactor->GetFrontierType(),
            max_queue_sizing, max_queue_sizing1),
            "BFS Problem Reset failed", __FILE__, __LINE__))
            return retval;

        if (retval = util::GRError(enactor->Reset(),
            "BFS Enactor Reset failed", __FILE__, __LINE__))
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
            "BFS Enact failed", __FILE__, __LINE__)) return retval;
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

    // compute reference CPU BFS solution for source-distance
    if (!quick_mode)
    {
        if (!quiet_mode)
        {
            printf("Computing reference value ...\n");
        }
        ReferenceBFS<VertexId, SizeT, Value,
            MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>(
            graph,
            reference_check_label,
            reference_check_preds,
            src,
            quiet_mode);
        if (!quiet_mode)
        {
            printf("\n");
        }
    }

    cpu_timer.Start();
    // copy out results
    if (retval = util::GRError(problem->Extract(h_labels, h_preds),
        "BFS Problem Extraction failed", __FILE__, __LINE__)) return retval;

    // verify the result
    if ((!quick_mode) && (!quiet_mode))
    {
        printf("Label Validity: ");
        int num_errors = CompareResults(
            h_labels, reference_check_label,
            graph->nodes, true, quiet_mode);
        if (num_errors > 0)
        {
            printf("%d errors occurred.", num_errors);
        }
        printf("\n");

        if (MARK_PREDECESSORS)
        {
            printf("Predecessor Validity: \n");
            num_errors = 0;
            #pragma omp parallel for
            for (VertexId v=0; v<graph->nodes; v++)
            {
                if (h_labels[v] ==
                    /*(ENABLE_IDEMPOTENCE ? -1 :*/ util::MaxValue<VertexId>())
                    continue; // unvisited vertex
                if (v == src && h_preds[v] == util::InvalidValue<VertexId>()) continue; // source vertex
                VertexId pred = h_preds[v];
                if (pred >= graph->nodes || pred < 0)
                {
                    if (num_errors == 0)
                        printf("INCORRECT: pred[%lld] : %lld out of bound\n",
                            (long long)v, (long long)pred);
                    #pragma omp atomic
                    num_errors ++;
                    continue;
                }
                if (h_labels[v] != h_labels[pred] + 1)
                {
                    if (num_errors == 0)
                        printf("INCORRECT: label[%lld] (%lld) != label[%lld] (%lld) + 1\n",
                            (long long)v, (long long)h_labels[v], (long long)pred, (long long)h_labels[pred]);
                    #pragma omp atomic
                    num_errors ++;
                    continue;
                }

                bool v_found = false;
                for (SizeT t = graph->row_offsets[pred]; t < graph->row_offsets[pred+1]; t++)
                if (v == graph->column_indices[t])
                {
                    v_found = true;
                    break;
                }
                if (!v_found)
                {
                    if (num_errors == 0)
                        printf("INCORRECT: Vertex %lld not in Vertex %lld's neighbor list\n",
                            (long long)v, (long long)pred);
                    #pragma omp atomic
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

    if (!quick_mode && TO_TRACK)
    {
        VertexId **v_ = NULL;
        if (num_gpus > 1)
        {
            v_ = new VertexId*[num_gpus];
            for (int gpu=0; gpu<num_gpus; gpu++)
            {
                v_[gpu] = new VertexId[graph->nodes];
                for (VertexId v=0; v<graph->nodes; v++)
                    v_[gpu][v] = -1;
                for (VertexId v=0; v<problem->sub_graphs[gpu].nodes; v++)
                    v_[gpu][problem->original_vertexes[gpu][v]] = v;
            }
        }
        util::Track_Results(graph, num_gpus, (VertexId)1, h_labels, reference_check_label,
            num_gpus > 1 ? problem->partition_tables[0] : NULL, v_);
        char file_name[512];
        sprintf(file_name, "./eval/error_dump/error_%lld_%d.txt", (long long)time(NULL), gpu_idx[0]);
        util::Output_Errors(file_name, graph -> nodes, num_gpus, (VertexId)0, h_labels, reference_check_label,
            num_gpus > 1 ? problem->partition_tables[0] : NULL, v_);
        if (num_gpus > 1)
        {
            for (int gpu=0; gpu<num_gpus; gpu++)
            {
                delete[] v_[gpu]; v_[gpu] = NULL;
            }
            delete[] v_; v_=NULL;
        }
    }

    // display Solution
    if (!quiet_mode)
    {
        DisplaySolution<VertexId, SizeT, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>
        (h_labels, h_preds, graph->nodes, quiet_mode);
    }

    info->ComputeTraversalStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), total_elapsed, h_labels);

    if (!quiet_mode)
    {
        Display_Memory_Usage(num_gpus, gpu_idx, org_size, problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
        Display_Performance_Profiling(enactor);
#endif
    }

    // Clean up
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
    if (reference_labels) {delete[] reference_labels; reference_labels = NULL;}
    if (reference_preds ) {delete[] reference_preds ; reference_preds  = NULL;}
    if (h_labels        ) {delete[] h_labels        ; h_labels         = NULL;}
    if (gpu_idx         ) {delete[] gpu_idx         ; gpu_idx          = NULL;}
    if (org_size        ) {delete[] org_size        ; org_size         = NULL;}
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();

    if (h_preds         )
    {
        if (info->info["output_filename"].get_str() != "")
        {
            cpu_timer.Start();
            std::ofstream fout;
            size_t buf_size = 1024 * 1024 * 16;
            char *fout_buf = new char[buf_size];
            fout.rdbuf() -> pubsetbuf(fout_buf, buf_size);
            fout.open(info->info["output_filename"].get_str().c_str());

            for (VertexId v=0; v<graph->nodes; v++)
            {
                if (v == src) fout<< v+1 << "," << v+1 << std::endl; // root node
                else if (h_preds[v] != -2) // valid pred
                    fout<< v+1 << "," << h_preds[v]+1 << std::endl;
            }

            fout.close();
            delete[] fout_buf; fout_buf = NULL;
            cpu_timer.Stop();
            info->info["write_time"] = cpu_timer.ElapsedMillis();
        }
        delete[] h_preds         ; h_preds          = NULL;
    }
    return retval;
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        MARK_PREDECESSORS >
cudaError_t RunTests_enable_idempotence(Info<VertexId, SizeT, Value> *info)
{
    if (info->info["idempotent"].get_bool())
        return RunTests <VertexId, SizeT, Value,/* INSTRUMENT, DEBUG, SIZE_CHECK,*/
                 MARK_PREDECESSORS, true > (info);
    else
        return RunTests <VertexId, SizeT, Value,/* INSTRUMENT, DEBUG, SIZE_CHECK,*/
                 MARK_PREDECESSORS, false> (info);
}

/**
 * @brief RunTests entry
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
    typename    VertexId,
    typename    SizeT,
    typename    Value>
cudaError_t RunTests_mark_predecessors(Info<VertexId, SizeT, Value> *info)
{
    if (info->info["mark_predecessors"].get_bool())
        return RunTests_enable_idempotence<VertexId, SizeT, Value, /*INSTRUMENT,
                                    DEBUG, SIZE_CHECK,*/  true> (info);
    else
        return RunTests_enable_idempotence<VertexId, SizeT, Value,/* INSTRUMENT,
                                    DEBUG, SIZE_CHECK,*/ false> (info);
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
    //typedef int VertexId;  // Use int as the vertex identifier
    //typedef int Value;     // Use int as the value type
    //typedef long long SizeT;     // Use int as the graph size type

    Csr<VertexId, SizeT, Value> csr(false);  // CSR graph we process on
    Csr<VertexId, SizeT, Value> csc(false);  // CSC graph we process on
    Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

    // graph construction or generation related parameters
    info->info["undirected"] = args -> CheckCmdLineFlag("undirected");

    cpu_timer2.Start();
    info->Init("BFS", *args, csr, csc);  // initialize Info structure
    cpu_timer2.Stop();
    info->info["load_time"] = cpu_timer2.ElapsedMillis();

    cudaError_t retval = RunTests_mark_predecessors<VertexId, SizeT, Value>(info);  // run test

    cpu_timer.Stop();
    info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!(info->info["quiet_mode"].get_bool()))
    {
        info->DisplayStats();  // display collected statistics
    }

    info->CollectInfo();  // collected all the info and put into JSON mObject
    if (info) {delete info; info=NULL;}
    return retval;
}

template <
    typename VertexId, // the vertex identifier type, usually int or long long
    typename SizeT   > // the size tyep, usually int or long long
int main_Value(CommandLineArgs *args)
{
    // Value = VertexId for bfs
    return main_<VertexId, SizeT, VertexId>(args);
//    if (args -> CheckCmdLineFlag("64bit-Value"))
//        return main_<VertexId, SizeT, long long>(args);
//    else
//        return main_<VertexId, SizeT, int      >(args);
}

template <
    typename VertexId>
int main_SizeT(CommandLineArgs *args)
{
// can be disabled to reduce compile time
    if (args -> CheckCmdLineFlag("64bit-SizeT") || sizeof(VertexId) > 4)
        return main_Value<VertexId, long long>(args);
    else
        return main_Value<VertexId, int      >(args);
}

int main_VertexId(CommandLineArgs *args)
{
// can be disabled to reduce compile time
// atomicMin(long long) is only available for compute capability 3.5 or higher
    if (args -> CheckCmdLineFlag("64bit-VertexId"))
//#if __GR_CUDA_ARCH__ <= 300
//    {
//        printf("64bit-VertexId disabled, because atomicMin(long long) is only supported by compute capability 3.5 or higher\n");
//        return 1;
//    }
//#else
        return main_SizeT<long long>(args);
//#endif
    else
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
