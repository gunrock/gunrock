// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file bfs_app.cu
 *
 * @brief Gunrock breadth-first search (BFS) application
 */

#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// breadth-first search includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

/**
 * @brief BFS_Parameter structure
 */
struct BFS_Parameter : gunrock::app::TestParameter_Base
{
public:
    bool   mark_predecessors ;  // mark src-distance vs. parent vertices
    bool   enable_idempotence;  // enable idempotence operation
    bool   direction_optimized; // enable direction optimization
    double max_queue_sizing1 ;  // maximum queue sizing factor

    BFS_Parameter()
    {
        mark_predecessors  = false;
        enable_idempotence = false;
        direction_optimized = false;
        max_queue_sizing1  = -1.0f;
    }

    ~BFS_Parameter()
    {
    }
};

template<typename VertexId, typename SizeT, typename Value,
         //bool INSTRUMENT, bool DEBUG, bool SIZE_CHECK,
         bool MARK_PREDECESSORS, bool ENABLE_IDEMPOTENCE>
float runBFS(GRGraph* output, BFS_Parameter *parameter);

/**
 * @brief Run test
 *
 * @tparam VertexId          Vertex identifier type
 * @tparam Value             Attribute type
 * @tparam SizeT             Graph size type
 * @tparam MARK_PREDECESSORS Enable mark predecessors
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 *
 * \return Elapsed run time in milliseconds
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        MARK_PREDECESSORS >
float RunTests_enable_idempotence(GRGraph* output, BFS_Parameter *parameter)
{
    if (parameter->enable_idempotence)
        return runBFS<VertexId, SizeT, Value,/*INSTRUMENT, DEBUG,
               SIZE_CHECK,*/ MARK_PREDECESSORS,  true>(output, parameter);
    else
        return runBFS<VertexId, SizeT, Value,/*INSTRUMENT, DEBUG,
               SIZE_CHECK,*/ MARK_PREDECESSORS, false>(output, parameter);
}

/**
 * @brief Run test
 *
 * @tparam VertexId   Vertex identifier type
 * @tparam Value      Attribute type
 * @tparam SizeT      Graph size type
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 *
 * \return Elapsed run time in milliseconds
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value>
float RunTests_mark_predecessors(GRGraph* output, BFS_Parameter *parameter)
{
    if (parameter->mark_predecessors)
        return RunTests_enable_idempotence
            <VertexId, SizeT, Value, true>(output, parameter);
    else
        return RunTests_enable_idempotence
            <VertexId, SizeT, Value, false>(output, parameter);
}

/**
 * @brief Run test
 *
 * @tparam VertexId           Vertex identifier type
 * @tparam Value              Attribute type
 * @tparam SizeT              Graph size type
 * @tparam MARK_PREDECESSORS  Enable mark predecessors
 * @tparam ENABLE_IDEMPOTENCE Enable idempotent operation
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 *
 * \return Elapsed run time in milliseconds
 */
template <
    typename    VertexId,
    typename    SizeT,
    typename    Value,
    bool        MARK_PREDECESSORS,
    bool        ENABLE_IDEMPOTENCE >
float runBFS(GRGraph* output, BFS_Parameter *parameter)
{
    typedef BFSProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS,
            ENABLE_IDEMPOTENCE>
            //(MARK_PREDECESSORS && ENABLE_IDEMPOTENCE) >
            Problem; // does not use double buffer

    typedef BFSEnactor < Problem>
            //INSTRUMENT,
            //DEBUG,
            //SIZE_CHECK >
            Enactor;

    Csr<VertexId, SizeT, Value> *graph =
        (Csr<VertexId, SizeT, Value>*)parameter->graph;
    bool          quiet                = parameter -> g_quiet;
    int           max_grid_size        = parameter -> max_grid_size;
    int           num_gpus             = parameter -> num_gpus;
    int           num_iters            = parameter -> iterations;
    double        max_queue_sizing     = parameter -> max_queue_sizing;
    double        max_queue_sizing1    = parameter -> max_queue_sizing1;
    double        max_in_sizing        = parameter -> max_in_sizing;
    ContextPtr   *context              = (ContextPtr*)parameter -> context;
    std::string   partition_method     = parameter -> partition_method;
    int          *gpu_idx              = parameter -> gpu_idx;
    cudaStream_t *streams              = parameter -> streams;
    float         partition_factor     = parameter -> partition_factor;
    int           partition_seed       = parameter -> partition_seed;
    bool          g_stream_from_host   = parameter -> g_stream_from_host;
    std::string   traversal_mode       = parameter -> traversal_mode;
    bool          instrument           = parameter -> instrumented;
    bool          debug                = parameter -> debug;
    bool          size_check           = parameter -> size_check;
    bool          undirected           = parameter -> g_undirected;
    bool          direction_optimized  = parameter -> direction_optimized;
    size_t       *org_size             = new size_t  [num_gpus];
    // Allocate host-side label array
    VertexId     *h_labels             = new VertexId[graph->nodes];
    VertexId     *h_preds              = NULL;

    if (MARK_PREDECESSORS)
    {
        h_preds = new VertexId[graph->nodes];
    }

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }
    Problem *problem = new Problem(direction_optimized, undirected);  // Allocate problem on GPU

    util::GRError( problem->Init(
        g_stream_from_host,
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
        "Problem BFS Initialization Failed", __FILE__, __LINE__);

    Enactor *enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check, direction_optimized);  // BFS enactor map
    util::GRError(
        enactor->Init(context, problem, max_grid_size, traversal_mode),
        "BFS Enactor init failed", __FILE__, __LINE__);

    CpuTimer cpu_timer;
    float elapsed = 0.0f;
    for (int i = 0; i < num_iters; ++i)
    {
        printf("Round %d of bfs.\n", i+1);
        util::GRError(
                problem->Reset(parameter->src[i], enactor->GetFrontierType(),
                    max_queue_sizing, max_queue_sizing1),
                "BFS Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(
                enactor->Reset(), "BFS Enactor Reset failed", __FILE__, __LINE__);

        cpu_timer.Start();

        util::GRError(
                enactor->Enact(parameter->src[i], traversal_mode),
                "BFS Problem Enact Failed", __FILE__, __LINE__);

        cpu_timer.Stop();

        elapsed += cpu_timer.ElapsedMillis();
    }

    // Copy out results
    util::GRError(
        problem->Extract(h_labels, h_preds),
        "BFS Problem Data Extraction Failed", __FILE__, __LINE__);

    output->node_value1 = (Value*)&h_labels[0];
    if (MARK_PREDECESSORS) output->node_value2 = (VertexId*)&h_preds[0];

    if (!quiet)
    {
        printf(" GPU Breath-First Search finished in %lf msec.\n", elapsed);
    }

    // Clean up
    if (org_size) delete[] org_size; org_size = NULL;
    if (enactor ) delete   enactor ; enactor  = NULL;
    if (problem ) delete   problem ; problem  = NULL;

    return elapsed;
}

/**
 * @brief Dispatch function to handle configurations
 *
 * @param[out] grapho  Pointer to output graph structure of the problem
 * @param[in]  graphi  Pointer to input graph we need to process on
 * @param[in]  config  Primitive-specific configurations
 * @param[in]  data_t  Data type configurations
 * @param[in]  context ModernGPU context
 * @param[in]  streams CUDA stream
 *
 * \return Elapsed run time in milliseconds
 */
float dispatch_bfs(
    GRGraph*       grapho,
    const GRGraph* graphi,
    const GRSetup* config,
    const GRTypes  data_t,
    ContextPtr*    context,
    cudaStream_t*  streams)
{
    BFS_Parameter *parameter = new BFS_Parameter;
    parameter->iterations = config -> num_iters;
    parameter->src = (long long*)malloc(sizeof(long long)*config -> num_iters);
    parameter->context  = context;
    parameter->streams  = streams;
    parameter->g_quiet  = config -> quiet;
    parameter->num_gpus = config -> num_devices;
    parameter->gpu_idx  = config -> device_list;
    parameter->mark_predecessors  = config -> mark_predecessors;
    parameter->enable_idempotence = config -> enable_idempotence;

    float elapsed_time;

    switch (data_t.VTXID_TYPE)
    {
    case VTXID_INT:
    {
        switch (data_t.SIZET_TYPE)
        {
        case SIZET_INT:
        {
            switch (data_t.VALUE_TYPE)
            {
            case VALUE_INT:    // template type = <int, int, int>
            {
                // build input CSR format graph
                Csr<int, int, int> csr(false);
                csr.nodes = graphi->num_nodes;
                csr.edges = graphi->num_edges;
                csr.row_offsets    = (int*)graphi->row_offsets;
                csr.column_indices = (int*)graphi->col_indices;
                parameter->graph = &csr;

                // determine source vertex to start
                switch (config -> source_mode)
                {
                case randomize:
                {
                    for (int i = 0; i < parameter->iterations; ++i)
                    {
                        parameter->src[i] = graphio::RandomNode(csr.nodes);
                    }
                    break;
                }
                case largest_degree:
                {
                    int max_deg = 0;
                    int node_id = csr.GetNodeWithHighestDegree(max_deg);
                    for (int i = 0; i < config -> num_iters; ++i)
                    {
                        parameter->src[i] = node_id;
                    }
                    break;
                }
                case manually:
                {
                    for (int i = 0; i < parameter->iterations; ++i)
                    {
                        parameter->src[i] = config -> source_vertex[i];
                    }
                    break;
                }
                default:
                {
                    for (int i = 0; i < parameter->iterations; ++i)
                    {
                        parameter->src[i] = 0;
                    }
                    break;
                }
                }
                if (!parameter->g_quiet)
                {
                    printf(" source: %lld", (long long) parameter->src[0]);
                    for (int i = 1; i < config -> num_iters; ++i)
                    {
                        printf(",%lld", (long long) parameter->src[i]);
                    }
                    printf("\n");
                }

                elapsed_time = RunTests_mark_predecessors<int, int, int>(grapho, parameter);

                // reset for free memory
                csr.row_offsets    = NULL;
                csr.column_indices = NULL;
                break;
            }
            case VALUE_UINT:    // template type = <int, uint, int>
            {
                // not yet support
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT:    // template type = <int, float, int>
            {
                // not yet support
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            }
            break;
        }
        }
        break;
    }
    }
    free(parameter->src);
    return elapsed_time;
}

/*
 * @brief Entry of gunrock_bfs function
 *
 * @param[out] grapho Pointer to output graph structure of the problem
 * @param[in]  graphi Pointer to input graph we need to process on
 * @param[in]  config Gunrock primitive specific configurations
 * @param[in]  data_t Gunrock data type structure
 */
float gunrock_bfs(
    GRGraph*       grapho,
    const GRGraph* graphi,
    const GRSetup* config,
    const GRTypes  data_t)
{
    // GPU-related configurations
    int           num_gpus =    0;
    int           *gpu_idx = NULL;
    ContextPtr    *context = NULL;
    cudaStream_t  *streams = NULL;

    num_gpus = config -> num_devices;
    gpu_idx  = new int [num_gpus];
    for (int i = 0; i < num_gpus; ++i)
    {
        gpu_idx[i] = config -> device_list[i];
    }

    // Create streams and MordernGPU context for each GPU
    streams = new cudaStream_t[num_gpus * num_gpus * 2];
    context = new ContextPtr[num_gpus * num_gpus];
    if (!config -> quiet) { printf(" using %d GPUs:", num_gpus); }
    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
        if (!config -> quiet) { printf(" %d ", gpu_idx[gpu]); }
        util::SetDevice(gpu_idx[gpu]);
        for (int i = 0; i < num_gpus * 2; ++i)
        {
            int _i = gpu * num_gpus * 2 + i;
            util::GRError(cudaStreamCreate(&streams[_i]),
                          "cudaStreamCreate fialed.", __FILE__, __LINE__);
            if (i < num_gpus)
            {
                context[gpu * num_gpus + i] =
                    mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu],
                                                       streams[_i]);
            }
        }
    }
    if (!config -> quiet) { printf("\n"); }

    return dispatch_bfs(grapho, graphi, config, data_t, context, streams);
}

/*
 * @brief Simple interface take in CSR arrays as input
 *
 * @param[out] bfs_label            Return BFS label (depth) per nodes or the predecessor per nodes
 * @param[in]  num_nodes            Number of nodes of the input graph
 * @param[in]  num_edges            Number of edges of the input graph
 * @param[in]  row_offsets          CSR-formatted graph input row offsets
 * @param[in]  col_indices          CSR-formatted graph input column indices
 * @param[in]  num_iters            Number of BFS runs. Note if num_iters > 1, the bfs_lbel will only store the results from the last run
 * @param[in]  source               Sources to begin traverse
 * @param[in]  source_mode          Enumerator of source mode: manually, randomize, largest_degree
 * @param[in]  mark_predecessors    If the flag is set, mark predecessors instead of bfs label
 * @param[in]  enable_idempotence   If the flag is set, use optimizations that allow idempotence operation (will usually bring better performance)
 */
float bfs(
    int*       bfs_label,
    int*       bfs_preds,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  num_iters,
    int* source,
    enum SrcMode source_mode,
    const bool mark_predecessors,
    const bool enable_idempotence)
{
    struct GRTypes data_t;          // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;  // integer vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;  // integer graph size type
    data_t.VALUE_TYPE = VALUE_INT;  // integer attributes type

    struct GRSetup* config = InitSetup(num_iters, source);  // primitive-specific configures
    config -> mark_predecessors  = mark_predecessors;    // do not mark predecessors
    config -> enable_idempotence = enable_idempotence;    // wether enable idempotence
    config -> source_mode = source_mode;

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graphi->num_nodes   = num_nodes;  // setting graph nodes
    graphi->num_edges   = num_edges;  // setting graph edges
    graphi->row_offsets = (void*)&row_offsets[0];  // setting row_offsets
    graphi->col_indices = (void*)&col_indices[0];  // setting col_indices


    float elapsed_time = gunrock_bfs(grapho, graphi, config, data_t);
    memcpy(bfs_label, (int*)grapho->node_value1, num_nodes * sizeof(int));
    if (mark_predecessors) 
        memcpy(bfs_preds, (int*)grapho->node_value2, num_nodes * sizeof(int));

    if (graphi) free(graphi);
    if (grapho) free(grapho);
    if (config) free(config);

    return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
