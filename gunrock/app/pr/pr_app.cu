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

/**
 * @brief PR_Parameter structure
 */
struct PR_Parameter : gunrock::app::TestParameter_Base
{
public:
    float    delta          ;  // Delta value for PageRank
    float    error          ;  // Error threshold PageRank
    int      max_iter       ;  // Maximum number of iteration
    bool     normalized     ;

    PR_Parameter()
    {
        delta    = 0.85f;
        error    = 0.01f;
        max_iter =    50;
        normalized = false;
    }

    ~PR_Parameter()
    {
    }
};

template <
    typename VertexId,
    typename SizeT,
    typename Value,
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK,
    bool NORMALIZED>
void runPageRank(GRGraph *output, PR_Parameter *parameter);

/**
 * @brief Run test
 *
 * @tparam VertexId   Vertex identifier type
 * @tparam Value      Attribute type
 * @tparam SizeT      Graph size type
 * @tparam INSTRUMENT Keep kernels statics
 * @tparam DEBUG      Keep debug statics
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 */
template <
    typename      VertexId,
    typename      SizeT,
    typename      Value>
    //bool          INSTRUMENT,
    //bool          DEBUG,
    //bool          SIZE_CHECK >
void normalizedPageRank(GRGraph *output, PR_Parameter *parameter)
{
    if (parameter -> normalized)
        runPageRank<VertexId, SizeT, Value, true > (output, parameter);
    else
        runPageRank<VertexId, SizeT, Value, false> (output, parameter);
}

/**
 * @brief Run test
 *
 * @tparam VertexId   Vertex identifier type
 * @tparam Value      Attribute type
 * @tparam SizeT      Graph size type
 * @tparam INSTRUMENT Keep kernels statics
 * @tparam DEBUG      Keep debug statics
 * @tparam SIZE_CHECK Enable size check
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK,
    bool NORMALIZED >
void runPageRank(GRGraph *output, PR_Parameter *parameter)
{
    typedef PRProblem < VertexId,
            SizeT,
            Value,
            NORMALIZED > Problem;

    typedef PREnactor < Problem>
            //INSTRUMENT,
            //DEBUG,
            //SIZE_CHECK > 
            Enactor;

    Csr<VertexId, SizeT, Value>
    *graph              = (Csr<VertexId, SizeT, Value>*)parameter->graph;
    bool          quiet              = parameter -> g_quiet;
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
    VertexId      src                = parameter -> src[0];
    Value         delta              = parameter -> delta;
    Value         error              = parameter -> error;
    SizeT         max_iter           = parameter -> max_iter;
    std::string   traversal_mode     = parameter -> traversal_mode;
    bool          instrument         = parameter -> instrumented;
    bool          debug              = parameter -> debug;
    bool          size_check         = parameter -> size_check;
    size_t       *org_size           = new size_t  [num_gpus];
    // Allocate host-side label arrays
    Value        *h_rank             = new Value   [graph->nodes];
    VertexId     *h_node_id          = new VertexId[graph->nodes];

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    Problem *problem = new Problem(false);  // Allocate problem on GPU
    util::GRError(
        problem->Init(
            g_stream_from_host,
            graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method,
            streams,
            context,
            max_queue_sizing,
            max_in_sizing,
            partition_factor,
            partition_seed),
        "PR Initialization Failed", __FILE__, __LINE__);

    Enactor* enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);  // enactor map
    util::GRError(
        enactor->Init(context, problem, traversal_mode, max_grid_size),
        "PR Enactor Init failed", __FILE__, __LINE__);

    // Perform PageRank
    CpuTimer cpu_timer;

    util::GRError(
        problem->Reset(src, delta, error, max_iter,
                       enactor->GetFrontierType(), max_queue_sizing),
        "PR Problem Data Reset Failed", __FILE__, __LINE__);
    util::GRError(
        enactor->Reset(), "PR Enactor Reset Reset failed", __FILE__, __LINE__);

    cpu_timer.Start();
    util::GRError(
        enactor->Enact(traversal_mode),
        "PR Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        problem->Extract(h_rank, h_node_id),
        "PR Problem Data Extraction Failed", __FILE__, __LINE__);

    double total_pr = 0;
    for (SizeT i = 0; i < graph->nodes; ++i)
    {
        total_pr += h_rank[i];
    }
    if (!quiet) { printf(" Total rank : %.8e\n", total_pr); }

    output->node_value1 = (Value*)&h_rank[0];
    output->node_value2 = (VertexId*)&h_node_id[0];

    if (!quiet) { printf(" GPU PageRank finished in %lf msec.\n", elapsed); }

    // Clean up
    if (org_size) { delete[] org_size; org_size = NULL; }
    if (problem ) { delete problem ; problem  = NULL; }
    if (enactor ) { delete enactor ; enactor  = NULL; }
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
 */
void dispatchPageRank(
    GRGraph       *grapho,
    const GRGraph *graphi,
    const GRSetup *config,
    const GRTypes  data_t,
    ContextPtr*    context,
    cudaStream_t*  streams)
{
    PR_Parameter *parameter = new PR_Parameter;
    parameter->src = (long long*)malloc(sizeof(long long));
    parameter->src[0] = -1;
    parameter->context      =  context;
    parameter->streams      =  streams;
    parameter->g_quiet      = config -> quiet;
    parameter->num_gpus     = config -> num_devices;
    parameter->gpu_idx      = config -> device_list;
    parameter->delta        = config -> pagerank_delta;
    parameter->error        = config -> pagerank_error;
    parameter->max_iter     = config -> max_iters;
    parameter->normalized   = config -> pagerank_normalized;
    parameter->g_undirected = true;

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
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT:    // template type = <int, uint, int>
            {
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT:    // template type = <int, float, int>
            {
                // build input csr format graph
                Csr<int, int, int> csr(false);
                csr.nodes = graphi->num_nodes;
                csr.edges = graphi->num_edges;
                csr.row_offsets    = (int*)graphi->row_offsets;
                csr.column_indices = (int*)graphi->col_indices;
                parameter->graph = &csr;

                normalizedPageRank<int, int, float>(grapho, parameter);

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
    free(parameter->src);
}

/*
 * @brief Entry of gunrock_pagerank function
 *
 * @param[out] grapho Pointer to output graph structure of the problem
 * @param[in]  graphi Pointer to input graph we need to process on
 * @param[in]  config Gunrock primitive specific configurations
 * @param[in]  data_t Gunrock data type structure
 */
void gunrock_pagerank(
    GRGraph       *grapho,
    const GRGraph *graphi,
    const GRSetup *config,
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

    dispatchPageRank(grapho, graphi, config, data_t, context, streams);
}

/*
 * @brief Simple interface take in CSR arrays as input
 *
 * @param[out] node_ids    Return top-ranked vertex IDs
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
    const int*          col_indices,
    bool                normalized)
{
    struct GRTypes data_t;            // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;    // integer vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;    // integer graph size type
    data_t.VALUE_TYPE = VALUE_FLOAT;  // float attributes type

    struct GRSetup *config = InitSetup(1, NULL);  // primitive-specific configures
    config -> top_nodes      = 10;           // number of top nodes
    config -> pagerank_normalized     = normalized;

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graphi->num_nodes   = num_nodes;  // setting graph nodes
    graphi->num_edges   = num_edges;  // setting graph edges
    graphi->row_offsets = (void*)&row_offsets[0];  // setting row_offsets
    graphi->col_indices = (void*)&col_indices[0];  // setting col_indices

    gunrock_pagerank(grapho, graphi, config, data_t);
    memcpy(pagerank, (float*)grapho->node_value1, num_nodes * sizeof(float));
    memcpy(node_ids, (  int*)grapho->node_value2, num_nodes * sizeof(  int));

    if (graphi) free(graphi);
    if (grapho) free(grapho);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
