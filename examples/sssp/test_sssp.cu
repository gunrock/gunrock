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
 * @brief Simple test driver program for single source shortest path.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>


#include <gunrock/app/sample/sample_enactor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <gunrock/util/shared_utils.cuh>

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
using namespace gunrock::app::sssp;

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
        "[--delta_factor=<factor>] Delta factor for delta-stepping SSSP.\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution (VertexId *source_path, SizeT num_nodes)
{
    if (num_nodes > 40) num_nodes = 40;

    printf("[");
    for (VertexId i = 0; i < num_nodes; ++i)
    {
        PrintValue(i);
        printf(":");
        PrintValue(source_path[i]);
        printf(" ");
    }
    printf("]\n");
}

/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference SSSP ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_values Host-side vector to store CPU computed labels for each node
 * @param[in] node_preds Host-side vector to store CPU computed predecessors for each node
 * @param[in] src Source node where SSSP starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool     MARK_PREDECESSORS >
void ReferenceSssp(
    const Csr<VertexId, SizeT, Value> &graph,
    Value                             *node_values,
    VertexId                          *node_preds,
    VertexId                          src,
    bool                              quiet)
{
    using namespace boost;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS, no_property,
            property <edge_weight_t, unsigned int> > Graph;

    typedef graph_traits<Graph>::vertex_descriptor vertex_descriptor;
    typedef graph_traits<Graph>::edge_descriptor edge_descriptor;

    typedef std::pair<VertexId, VertexId> Edge;

    Edge   *edges = ( Edge*)malloc(sizeof( Edge) * graph.edges);
    Value *weight = (Value*)malloc(sizeof(Value) * graph.edges);

    for (SizeT i = 0; i < graph.nodes; ++i)
    {
        for (SizeT j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
        {
            edges[j] = Edge(i, graph.column_indices[j]);
            weight[j] = graph.edge_values[j];
        }
    }

    Graph g(edges, edges + graph.edges, weight, graph.nodes);

    std::vector<Value> d(graph.nodes);
    std::vector<vertex_descriptor> p(graph.nodes);
    vertex_descriptor s = vertex(src, g);

    property_map<Graph, vertex_index_t>::type indexmap = get(vertex_index, g);

    //
    // Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    if (MARK_PREDECESSORS)
    {
        dijkstra_shortest_paths(g, s,
            predecessor_map(boost::make_iterator_property_map(
                p.begin(), get(boost::vertex_index, g))).distance_map(
                    boost::make_iterator_property_map(
                        d.begin(), get(boost::vertex_index, g))));
    }
    else
    {
        dijkstra_shortest_paths(g, s,
            distance_map(boost::make_iterator_property_map(
                d.begin(), get(boost::vertex_index, g))));
    }
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    if (!quiet) { printf("CPU SSSP finished in %lf msec.\n", elapsed); }

    Coo<Value, Value>* sort_dist = NULL;
    Coo<VertexId, VertexId>* sort_pred = NULL;
    sort_dist = (Coo<Value, Value>*)malloc(
                    sizeof(Coo<Value, Value>) * graph.nodes);
    if (MARK_PREDECESSORS)
    {
        sort_pred = (Coo<VertexId, VertexId>*)malloc(
                        sizeof(Coo<VertexId, VertexId>) * graph.nodes);
    }
    graph_traits < Graph >::vertex_iterator vi, vend;
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    {
        sort_dist[(*vi)].row = (*vi);
        sort_dist[(*vi)].col = d[(*vi)];
    }
    std::stable_sort(
        sort_dist, sort_dist + graph.nodes,
        RowFirstTupleCompare<Coo<Value, Value> >);

    if (MARK_PREDECESSORS)
    {
        for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
        {
            sort_pred[(*vi)].row = (*vi);
            sort_pred[(*vi)].col = p[(*vi)];
        }
        std::stable_sort(
            sort_pred, sort_pred + graph.nodes,
            RowFirstTupleCompare< Coo<VertexId, VertexId> >);
    }

    for (SizeT i = 0; i < graph.nodes; ++i)
    {
        node_values[i] = sort_dist[i].col;
    }
    if (MARK_PREDECESSORS)
    {
        for (SizeT i = 0; i < graph.nodes; ++i)
        {
            node_preds[i] = sort_pred[i].col;
        }
    }
    if (sort_dist) free(sort_dist);
    if (sort_pred) free(sort_pred);
}


/**
 * @brief Run SSSP tests
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
    typename VertexId,
    typename SizeT,
    typename Value,
    bool MARK_PREDECESSORS >
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info)
{
    typedef SSSPProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS > Problem;

    typedef SSSPEnactor < Problem > Enactor;

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
    int         delta_factor        = info->info["delta_factor"     ].get_int  ();
    std::string src_type            = info->info["source_type"      ].get_str  ();
    int      src_seed               = info->info["source_seed"      ].get_int  ();
    int      communicate_latency    = info->info["communicate_latency"].get_int ();
    float    communicate_multipy    = info->info["communicate_multipy"].get_real();
    int      expand_latency         = info->info["expand_latency"    ].get_int ();
    int      subqueue_latency       = info->info["subqueue_latency"  ].get_int ();
    int      fullqueue_latency      = info->info["fullqueue_latency" ].get_int ();
    int      makeout_latency        = info->info["makeout_latency"   ].get_int ();
    if (max_queue_sizing < 1.2) max_queue_sizing=1.2;
    if (max_in_sizing < 0) max_in_sizing = 1.0;
    if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;

    CpuTimer    cpu_timer;
    cudaError_t retval              = cudaSuccess;

    cpu_timer.Start();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr   *context = (ContextPtr*)  info->context;
    cudaStream_t *streams = (cudaStream_t*)info->streams;

    // Allocate host-side array (for both reference and GPU-computed results)
    Value    *reference_labels      = new Value[graph->nodes];
    Value    *h_labels              = new Value[graph->nodes];
    Value    *reference_check_label = (quick_mode) ? NULL : reference_labels;
    VertexId *reference_preds       = MARK_PREDECESSORS ? new VertexId[graph->nodes] : NULL;
    VertexId *h_preds               = MARK_PREDECESSORS ? new VertexId[graph->nodes] : NULL;
    VertexId *reference_check_pred  = (quick_mode || !MARK_PREDECESSORS) ? NULL : reference_preds;

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
        delta_factor,
        max_queue_sizing,
        max_in_sizing,
        partition_factor,
        partition_seed),
        "SSSP Problem Init failed", __FILE__, __LINE__))
        return retval;

    // Allocate SSSP enactor map
    Enactor* enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);
    if (retval = util::GRError(enactor->Init(
        context, problem, max_grid_size, traversal_mode),
        "SSSP Enactor Init failed", __FILE__, __LINE__))
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

    // perform SSSP
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
            src, enactor->GetFrontierType(),
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
        ReferenceSssp<VertexId, SizeT, Value, MARK_PREDECESSORS>(
            *graph,
            reference_check_label,
            reference_check_pred,
            src,
            quiet_mode);
        if (!quiet_mode) { printf("\n"); }
    }

    cpu_timer.Start();
    // Copy out results
    if (retval = util::GRError(problem->Extract(h_labels, h_preds),
        "SSSP Problem Data Extraction Failed", __FILE__, __LINE__))
        return retval;

    if (!quick_mode) {
        for (SizeT i = 0; i < graph->nodes; i++)
        {
            if (reference_check_label[i] == -1)
            {
                reference_check_label[i] = util::MaxValue<Value>();
            }
        }
    }

    if (!quiet_mode)
    {
        // Display Solution
        printf("\nFirst 40 labels of the GPU result.\n");
        DisplaySolution(h_labels, graph->nodes);
    }
    // Verify the result
    if (!quick_mode)
    {
        if (!quiet_mode) { printf("Label Validity: "); }
        int error_num = CompareResults(
                            h_labels, reference_check_label,
                            graph->nodes, true, quiet_mode);
        if (error_num > 0)
        {
            if (!quiet_mode) { printf("%d errors occurred.\n", error_num); }
        }
        if (!quiet_mode)
        {
            printf("\nFirst 40 labels of the reference CPU result.\n");
            DisplaySolution(reference_check_label, graph->nodes);
        }
    }

    info->ComputeTraversalStats(  // compute running statistics
        enactor->enactor_stats.GetPointer(), total_elapsed, h_labels);

    if (!quiet_mode)
    {
        if (MARK_PREDECESSORS)
        {
            printf("\nFirst 40 preds of the GPU result.\n");
            DisplaySolution(h_preds, graph->nodes);
            if (reference_check_label != NULL)
            {
                printf("\nFirst 40 preds of the reference CPU result (could be different because the paths are not unique).\n");
                DisplaySolution(reference_check_pred, graph->nodes);
            }
        }

        /*printf("\n\tMemory Usage(B)\t");
        for (int gpu = 0; gpu < num_gpus; gpu++)
            if (num_gpus > 1) {if (gpu != 0) printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1", gpu, gpu, gpu, gpu); else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);}
            else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
        if (num_gpus > 1) printf(" #keys%d", num_gpus);
        printf("\n");
        double max_queue_sizing_[2] = {0, 0}, max_in_sizing_ = 0;
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
                    if (factor > max_queue_sizing_[j]) max_queue_sizing_[j] = factor;
                }
                if (num_gpus > 1 && i != 0 )
                    for (int t = 0; t < 2; t++)
                    {
                        SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                        printf("\t %lld", (long long) x);
                        double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
                        if (factor > max_in_sizing_) max_in_sizing_ = factor;
                    }
            }
            if (num_gpus > 1) printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
            printf("\n");
        }
        printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
        if (num_gpus > 1) printf("\t in_sizing =\t %lf", max_in_sizing_);
        printf("\n");
        */
    }

    if (!quiet_mode)
    {
        Display_Memory_Usage(num_gpus, gpu_idx, org_size, problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
        Display_Performance_Profiling(enactor);
#endif
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
    if (reference_labels) {delete[] reference_labels; reference_labels = NULL;}
    if (h_labels        ) {delete[] h_labels        ; h_labels         = NULL;}
    if (reference_preds ) {delete[] reference_preds ; reference_preds  = NULL;}
    if (h_preds         ) {delete[] h_preds         ; h_preds          = NULL;}
    if (gpu_idx         ) {delete[] gpu_idx         ; gpu_idx          = NULL;}
    cpu_timer.Stop();
    info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    return retval;
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
        return RunTests<VertexId, SizeT, Value, /*INSTRUMENT,
                 DEBUG, SIZE_CHECK,*/ true>(info);
    else
        return RunTests<VertexId, SizeT, Value, /*INSTRUMENT,
                 DEBUG, SIZE_CHECK,*/ false>(info);
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

    // force edge values to be 1, don't enable this unless you really want to
    //for (SizeT e=0; e < csr.edges; e++)
    //    csr.edge_values[e] = 1;
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
