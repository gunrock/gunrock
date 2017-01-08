// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file bc_app.cu
 *
 * @brief Gunrock betweeness centrality (BC) application
 */

#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// betweeness centrality includes
#include <gunrock/app/bc/bc_enactor.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bc;

/**
 * @brief BC_Parameter structure
 */
struct BC_Parameter : gunrock::app::TestParameter_Base
{
public:
    std::string ref_filename;
    double max_queue_sizing1;

    BC_Parameter()
    {
        ref_filename = "";
        max_queue_sizing1 = -1.0;
    }

    ~BC_Parameter()
    {
    }
};

/**
 * @brief Graph edge properties (bundled properties)
 */
struct EdgeProperties
{
    int weight;
};

template <
    typename VertexId,
    typename SizeT,
    typename Value>
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK >
void runBC(GRGraph* output, BC_Parameter *parameter);

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
    typename Value>
    //bool INSTRUMENT,
    //bool DEBUG,
    //bool SIZE_CHECK >
void runBC(GRGraph* output, BC_Parameter *parameter)
{
    typedef BCProblem <VertexId,
            SizeT,
            Value,
            true>               // MARK_PREDECESSORS
            Problem;  // Does not use double buffer

    typedef BCEnactor <Problem>
            //INSTRUMENT,
            //DEBUG,
            //SIZE_CHECK >
            Enactor;

    Csr<VertexId, SizeT, Value> *graph =
        (Csr<VertexId, SizeT, Value>*)parameter->graph;
    bool          quiet              = parameter -> g_quiet;
    VertexId      src                = (VertexId)parameter -> src[0];
    int           max_grid_size      = parameter -> max_grid_size;
    int           num_gpus           = parameter -> num_gpus;
    double        max_queue_sizing   = parameter -> max_queue_sizing;
    double        max_queue_sizing1  = parameter -> max_queue_sizing1;
    double        max_in_sizing      = parameter -> max_in_sizing;
    ContextPtr   *context            = (ContextPtr*)parameter -> context;
    std::string   partition_method   = parameter -> partition_method;
    int          *gpu_idx            = parameter -> gpu_idx;
    cudaStream_t *streams            = parameter -> streams;
    float         partition_factor   = parameter -> partition_factor;
    int           partition_seed     = parameter -> partition_seed;
    bool          g_stream_from_host = parameter -> g_stream_from_host;
    bool          instrument         = parameter -> instrumented;
    bool          debug              = parameter -> debug;
    bool          size_check         = parameter -> size_check;
    size_t       *org_size           = new size_t  [num_gpus];
    // Allocate host-side arrays
    Value        *h_sigmas           = new Value   [graph->nodes];
    Value        *h_bc_values        = new Value   [graph->nodes];
    VertexId     *h_labels           = new VertexId[graph->nodes];

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    Problem* problem = new Problem(false);  // Allocate problem on GPU
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
        "BC Problem Initialization Failed", __FILE__, __LINE__);
    
    Enactor* enactor = new Enactor(
        num_gpus, gpu_idx, instrument, debug, size_check);  // BC enactor map
    util::GRError(
        enactor->Init(context, problem, max_grid_size),
        "BC Enactor init failed", __FILE__, __LINE__);

    // Perform BC
    CpuTimer cpu_timer;
    VertexId start_src;
    VertexId end_src;

    if (src == -1)
    {
        start_src = 0;
        end_src = graph->nodes;
    }
    else
    {
        start_src = src;
        end_src = src + 1;
    }

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        util::SetDevice(gpu_idx[gpu]);
        util::MemsetKernel <<< 128, 128 >>> (
            problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE),
            (Value)0.0f, (int)(problem->sub_graphs[gpu].nodes));
    }
    util::GRError(
        problem->Reset(0, enactor->GetFrontierType(),
                       max_queue_sizing, max_queue_sizing1),
        "BC Problem Data Reset Failed", __FILE__, __LINE__);

    cpu_timer.Start();
    for (VertexId i = start_src; i < end_src; ++i)
    {
        util::GRError(
            problem->Reset(i, enactor->GetFrontierType(),
                           max_queue_sizing, max_queue_sizing1),
            "BC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(
            enactor ->Reset(), "BC Enactor Reset failed", __FILE__, __LINE__);
        util::GRError(
            enactor ->Enact(i), "BC Problem Enact Failed", __FILE__, __LINE__);
    }

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        util::SetDevice(gpu_idx[gpu]);
        util::MemsetScaleKernel <<< 128, 128 >>> (
            problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE),
            (Value)0.5f, (int)(problem->sub_graphs[gpu].nodes));
    }
    cpu_timer.Stop();

    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        problem->Extract(h_sigmas, h_bc_values, h_labels),
        "BC Problem Data Extraction Failed", __FILE__, __LINE__);

    output->node_value1 = (Value*)&h_bc_values[0];

    if (!quiet)
    {
        printf(" GPU Betweenness Centrality finished in %lf msec.\n", elapsed);
    }

    // Clean up
    if (org_size) { delete[] org_size; org_size = NULL; }
    if (problem ) { delete   problem ; problem  = NULL; }
    if (enactor ) { delete   enactor ; enactor  = NULL; }
    if (h_sigmas) { delete[] h_sigmas; h_sigmas = NULL; }
    if (h_labels) { delete[] h_labels; h_labels = NULL; }
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
void dispatchBC(
    GRGraph*        grapho,
    const GRGraph*  graphi,
    const GRSetup*  config,
    const GRTypes   data_t,
    ContextPtr*     context,
    cudaStream_t*   streams)
{
    BC_Parameter* parameter = new BC_Parameter;
    parameter->src = (long long*)malloc(sizeof(long long));
    parameter->g_quiet  = config -> quiet;
    parameter->context  = context;
    parameter->streams  = streams;
    parameter->num_gpus = config -> num_devices;
    parameter->gpu_idx  = config -> device_list;

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
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT:    // template type = <int, uint, int>
            {
                // not support yet
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

                // determine source vertex to start
                switch (config -> source_mode)
                {
                case randomize:
                {
                    parameter->src[0] = graphio::RandomNode(csr.nodes);
                    break;
                }
                case largest_degree:
                {
                    int max_deg = 0;
                    parameter->src[0] = csr.GetNodeWithHighestDegree(max_deg);
                    break;
                }
                case manually:
                {
                    parameter->src[0] = config -> source_vertex[0];
                    break;
                }
                default:
                {
                    parameter->src[0] = 0;
                    break;
                }
                }
                if (!parameter->g_quiet)
                {
                    printf(" source: %lld\n", (long long) parameter->src[0]);
                }
                runBC<int, int, float>(grapho, parameter);

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
 * @brief Entry of gunrock_bc function
 *
 * @param[out] grapho Pointer to output graph structure of the problem
 * @param[in]  graphi Pointer to input graph we need to process on
 * @param[in]  config Gunrock primitive specific configurations
 * @param[in]  data_t Gunrock data type structure
 */
void gunrock_bc(
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

    dispatchBC(grapho, graphi, config, data_t, context, streams);
}

/*
 * @brief Simple interface take in CSR arrays as input
 *
 * @param[out] bc_scores   Return BC node centrality per nodes
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse/computation
 */
void bc(
    float*     bc_scores,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  source)
{
    struct GRTypes data_t;            // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;    // integer vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;    // integer graph size type
    data_t.VALUE_TYPE = VALUE_FLOAT;  // float attributes type

    struct GRSetup *config = InitSetup(1, NULL);  // primitive-specific configures
    config -> source_vertex[0] = source;        // source vertex to start

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graphi->num_nodes   = num_nodes;  // setting graph nodes
    graphi->num_edges   = num_edges;  // setting graph edges
    graphi->row_offsets = (void*)&row_offsets[0];  // setting row_offsets
    graphi->col_indices = (void*)&col_indices[0];  // setting col_indices

    gunrock_bc(grapho, graphi, config, data_t);
    memcpy(bc_scores, (float*)grapho->node_value1, num_nodes * sizeof(float));

    if (graphi) free(graphi);
    if (grapho) free(grapho);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
