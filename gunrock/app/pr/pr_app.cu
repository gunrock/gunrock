// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file pr_app.cu
 *
 * @brief Gunrock PageRank application
 */

#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// page-rank includes
#include <gunrock/app/pr/pr_enactor.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/app/pr/pr_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::pr;

struct Test_Parameter : gunrock::app::TestParameter_Base {
  public:
    float    delta          ;  // Delta value for PageRank
    float    error          ;  // Error threshold PageRank
    int      max_iter       ;  // Maximum number of iteration

    Test_Parameter() {
        delta    = 0.85f;
        error    = 0.01f;
        max_iter =    50;
        src      =    -1;
    }
    ~Test_Parameter() {
    }
};

template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK >
void runPageRank(GRGraph *output, Test_Parameter *parameter);

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG >
void sizeCheckPageRank(GRGraph *output, Test_Parameter *parameter) {
    if (parameter->size_check)
        runPageRank<VertexId, Value, SizeT, INSTRUMENT, DEBUG,
                    true > (output, parameter);
    else
        runPageRank<VertexId, Value, SizeT, INSTRUMENT, DEBUG,
                    false> (output, parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT >
void debugPageRank(GRGraph *output, Test_Parameter *parameter) {
    if (parameter->debug)
        sizeCheckPageRank<VertexId, Value, SizeT, INSTRUMENT,
                          true > (output, parameter);
    else
        sizeCheckPageRank<VertexId, Value, SizeT, INSTRUMENT,
                          false> (output, parameter);
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void runPageRank(GRGraph *output, Test_Parameter* parameter) {
    if (parameter->instrumented)
        debugPageRank<VertexId, Value, SizeT,  true>(output, parameter);
    else
        debugPageRank<VertexId, Value, SizeT, false>(output, parameter);
}

/**
 * @brief Run PR tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] src Source node for personalized PageRank (if any)
 * @param[in] delta Delta value for computing PageRank, usually set to .85
 * @param[in] error Error threshold value
 * @param[in] max_iter Max iteration for Page Rank computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] iterations Number of iterations for running the test
 * @param[in] context CudaContext for moderngpu to use
 *
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK >
void runPageRank(GRGraph *output, Test_Parameter *parameter) {
    typedef PRProblem < VertexId,
            SizeT,
            Value > PrProblem;

    typedef PREnactor < PrProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK > PrEnactor;

    Csr<VertexId, Value, SizeT>
    *graph              = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    int           max_grid_size      = parameter -> max_grid_size;
    int           num_gpus           = parameter -> num_gpus;
    double        max_queue_sizing   = parameter -> max_queue_sizing;
    double        max_in_sizing      = parameter -> max_in_sizing;
    ContextPtr   *context            = (ContextPtr*)parameter -> context;
    std::string   partition_method   = parameter -> partition_method;
    int          *gpu_idx            = parameter -> gpu_idx;
    cudaStream_t *streams            = parameter -> streams;
    float         partition_factor   = parameter -> partition_factor;
    int           partition_seed     = parameter -> partition_seed;
    bool          g_stream_from_host = parameter -> g_stream_from_host;
    VertexId      src                = parameter -> src;
    Value         delta              = parameter -> delta;
    Value         error              = parameter -> error;
    SizeT         max_iter           = parameter -> max_iter;
    int           traversal_mode     = parameter -> traversal_mode;
    size_t       *org_size           = new size_t  [num_gpus];
    // Allocate host-side label arrays
    Value        *h_rank             = new Value   [graph->nodes];
    VertexId     *h_node_id          = new VertexId[graph->nodes];

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    PrEnactor* enactor = new PrEnactor(num_gpus, gpu_idx);  // enactor map
    PrProblem *problem = new PrProblem;  // Allocate problem on GPU

    util::GRError(
        problem->Init(
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
        "Problem pr Initialization Failed", __FILE__, __LINE__);
    util::GRError(
        enactor->Init(context, problem, traversal_mode, max_grid_size),
        "PR Enactor Init failed", __FILE__, __LINE__);

    // Perform PageRank
    CpuTimer cpu_timer;

    util::GRError(
        problem->Reset(src, delta, error, max_iter,
                       enactor->GetFrontierType(), max_queue_sizing),
        "pr Problem Data Reset Failed", __FILE__, __LINE__);
    util::GRError(
        enactor->Reset(), "PR Enactor Reset Reset failed", __FILE__, __LINE__);

    printf("_________________________________________\n"); fflush(stdout);
    cpu_timer.Start();
    util::GRError(
        enactor->Enact(traversal_mode),
        "PR Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    printf("-----------------------------------------\n"); fflush(stdout);
    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        problem->Extract(h_rank, h_node_id),
        "PageRank Problem Data Extraction Failed", __FILE__, __LINE__);

    float total_pr = 0;
    for (int i = 0; i < graph->nodes; ++i) {
        total_pr += h_rank[i];
    }
    printf(" Total rank : %lf\n", total_pr);

    output->node_value1 = (Value*)&h_rank[0];
    output->node_value2 = (VertexId*)&h_node_id[0];

    printf(" GPU PageRank finished in %lf msec.\n", elapsed);

    // Clean up
    if (org_size) { delete org_size; org_size = NULL; }
    if (problem ) { delete problem ; problem  = NULL; }
    if (enactor ) { delete enactor ; enactor  = NULL; }
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] graph_o    output of pr problem
 * @param[out] node_ids   output of pr problem
 * @param[out] page_rank  output of pr problem
 * @param[in]  graph_i    GRGraph type input graph
 * @param[in]  config     specific configurations
 * @param[in]  data_t     data type configurations
 * @param[in]  context    moderngpu context
 */
void dispatchPageRank(
    GRGraph       *graph_o,
    const GRGraph *graph_i,
    const GRSetup  config,
    const GRTypes  data_t,
    ContextPtr*    context,
    cudaStream_t*  streams) {
    Test_Parameter *parameter = new Test_Parameter;
    parameter->context      =  context;
    parameter->streams      =  streams;
    parameter->num_gpus     = config.num_devices;
    parameter->gpu_idx      = config.device_list;
    parameter->delta        = config.pagerank_delta;
    parameter->error        = config.pagerank_error;
    parameter->max_iter     = config.max_iters;
    parameter->g_undirected = true;

    switch (data_t.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_t.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_t.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT: {  // template type = <int, uint, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {  // template type = <int, float, int>
                // build input csr format graph
                Csr<int, int, int> csr(false);
                csr.nodes = graph_i->num_nodes;
                csr.edges = graph_i->num_edges;
                csr.row_offsets    = (int*)graph_i->row_offsets;
                csr.column_indices = (int*)graph_i->col_indices;
                parameter->graph = &csr;

                runPageRank<int, float, int>(graph_o, parameter);

                // reset for free memory
                csr.row_offsets    = NULL;
                csr.column_indices = NULL;
                break;
            }
            }
            break;
        }
        }
        break;
    }
    }
}

/**
 * @brief run_pr entry
 *
 * @param[out] graph_o    output of pr problem
 * @param[out] node_ids   output of pr problem
 * @param[out] page_rank  output of pr problem
 * @param[in]  graph_i    input graph need to process on
 * @param[in]  config     gunrock primitive specific configurations
 * @param[in]  data_t     gunrock data_t struct
 */
void gunrock_pagerank(
    GRGraph       *graph_o,
    const GRGraph *graph_i,
    const GRSetup  config,
    const GRTypes  data_t) {
    // GPU-related configurations
    int           num_gpus =    0;
    int           *gpu_idx = NULL;
    ContextPtr    *context = NULL;
    cudaStream_t  *streams = NULL;

    num_gpus = config.num_devices;
    gpu_idx  = new int [num_gpus];
    for (int i = 0; i < num_gpus; ++i) {
        gpu_idx[i] = config.device_list[i];
    }

    // Create streams and MordernGPU context for each GPU
    streams = new cudaStream_t[num_gpus * num_gpus * 2];
    context = new ContextPtr[num_gpus * num_gpus];
    printf(" using %d GPUs:", num_gpus);
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        printf(" %d ", gpu_idx[gpu]);
        util::SetDevice(gpu_idx[gpu]);
        for (int i = 0; i < num_gpus * 2; ++i) {
            int _i = gpu * num_gpus * 2 + i;
            util::GRError(cudaStreamCreate(&streams[_i]),
                          "cudaStreamCreate fialed.", __FILE__, __LINE__);
            if (i < num_gpus) {
                context[gpu * num_gpus + i] =
                    mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu],
                                                       streams[_i]);
            }
        }
    }
    printf("\n");

    dispatchPageRank(graph_o, graph_i, config, data_t, context, streams);
}

/*
 * @brief Simple interface take in CSR arrays as input
 * @param[out] pagerank    Return PageRank scores per node
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse
 */
void pagerank(
    int*                node_ids,
    float*              pagerank,
    const int           num_nodes,
    const int           num_edges,
    const int*          row_offsets,
    const int*          col_indices) {
    struct GRTypes data_t;            // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;    // integer
    data_t.SIZET_TYPE = SIZET_INT;    // integer
    data_t.VALUE_TYPE = VALUE_FLOAT;  // float ranks

    struct GRSetup config;            // primitive-specific configures
    int list[] = {0, 1, 2, 3};        // device to run algorithm
    config.num_devices = sizeof(list) / sizeof(list[0]);  // number of devices
    config.device_list    =  list;    // device list to run algorithm
    config.pagerank_delta = 0.85f;    // default delta value
    config.pagerank_error = 0.01f;    // default error threshold
    config.max_iters      =    50;    // maximum number of iterations
    config.top_nodes      =    10;    // number of top nodes

    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    printf(" loaded %d nodes and %d edges\n", num_nodes, num_edges);

    gunrock_pagerank(graph_o, graph_i, config, data_t);
    memcpy(pagerank, (float*)graph_o->node_value1, num_nodes * sizeof(float));
    memcpy(node_ids, (  int*)graph_o->node_value2, num_nodes * sizeof(  int));

    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
