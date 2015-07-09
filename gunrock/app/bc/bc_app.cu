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

struct Test_Parameter : gunrock::app::TestParameter_Base {
  public:
    std::string ref_filename;
    double max_queue_sizing1;

    Test_Parameter() {
        ref_filename = "";
        max_queue_sizing1 = -1.0;
    }

    ~Test_Parameter() {
    }
};

/**
 * @brief Graph edge properties (bundled properties)
 */
struct EdgeProperties {
    int weight;
};

template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK >
void RunTests(GRGraph* output, Test_Parameter *parameter);

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG >
void RunTests_size_check(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->size_check)
        RunTests<VertexId, Value, SizeT, INSTRUMENT,
                 DEBUG,  true>(output, parameter);
    else
        RunTests<VertexId, Value, SizeT, INSTRUMENT,
                 DEBUG, false>(output, parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT >
void RunTests_debug(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->debug)
        RunTests_size_check<VertexId, Value, SizeT,
                            INSTRUMENT,  true>(output, parameter);
    else
        RunTests_size_check<VertexId, Value, SizeT,
                            INSTRUMENT, false> (output, parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT >
void RunTests_instrumented(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->instrumented)
        RunTests_debug<VertexId, Value, SizeT,  true>(output, parameter);
    else
        RunTests_debug<VertexId, Value, SizeT, false>(output, parameter);
}

/**
 * @brief Run betweenness centrality tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph object defined in main driver
 * @param[in] src
 * @param[in] ref_filename
 * @param[in] max_grid_size
 * @param[in] num_gpus
 * @param[in] max_queue_sizing
 * @param[in] iterations Number of iterations for running the test
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK >
void RunTests(GRGraph* output, Test_Parameter *parameter) {
    typedef BCProblem <VertexId,
            SizeT,
            Value,
            true,               // MARK_PREDECESSORS
            false > BcProblem;  // Does not use double buffer

    typedef BCEnactor <BcProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK > BcEnactor;

    Csr<VertexId, Value, SizeT> *graph =
        (Csr<VertexId, Value, SizeT>*)parameter->graph;
    VertexId      src                = (VertexId)parameter -> src;
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
    size_t       *org_size           = new size_t  [num_gpus];
    // Allocate host-side arrays
    Value        *h_sigmas           = new Value   [graph->nodes];
    Value        *h_bc_values        = new Value   [graph->nodes];
    Value        *h_ebc_values       = new Value   [graph->edges];
    VertexId     *h_labels           = new VertexId[graph->nodes];

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    BcEnactor* enactor = new BcEnactor(num_gpus, gpu_idx);  // BC enactor map
    BcProblem* problem = new BcProblem;  // Allocate problem on GPU

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
    util::GRError(
        enactor->Init(context, problem, max_grid_size),
        "BC Enactor init failed", __FILE__, __LINE__);

    // Perform BC
    CpuTimer cpu_timer;
    VertexId start_src;
    VertexId end_src;

    if (src == -1) {
        start_src = 0;
        end_src = graph->nodes;
    } else {
        start_src = src;
        end_src = src + 1;
    }

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        util::SetDevice(gpu_idx[gpu]);
        util::MemsetKernel <<< 128, 128 >>> (
            problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE),
            (Value)0.0f, (int)(problem->sub_graphs[gpu].nodes));
    }
    util::GRError(
        problem->Reset(0, enactor->GetFrontierType(),
                       max_queue_sizing, max_queue_sizing1),
        "BC Problem Data Reset Failed", __FILE__, __LINE__);

    printf("__________________________\n"); fflush(stdout);
    cpu_timer.Start();
    for (VertexId i = start_src; i < end_src; ++i) {
        util::GRError(
            problem->Reset(i, enactor->GetFrontierType(),
                           max_queue_sizing, max_queue_sizing1),
            "BC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(
            enactor ->Reset(), "BC Enactor Reset failed", __FILE__, __LINE__);
        util::GRError(
            enactor ->Enact(i), "BC Problem Enact Failed", __FILE__, __LINE__);
    }

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        util::SetDevice(gpu_idx[gpu]);
        util::MemsetScaleKernel <<< 128, 128 >>> (
            problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE),
            (Value)0.5f, (int)(problem->sub_graphs[gpu].nodes));
    }
    cpu_timer.Stop();
    printf("--------------------------\n"); fflush(stdout);
    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        problem->Extract(h_sigmas, h_bc_values, h_ebc_values, h_labels),
        "BC Problem Data Extraction Failed", __FILE__, __LINE__);

    output->node_value1 = (Value*)&h_bc_values[0];
    output->edge_value1 = (Value*)&h_ebc_values[0];

    printf("GPU BC finished in %lf msec.\n", elapsed);

    // Clean up
    if (org_size    ) { delete[] org_size    ; org_size     = NULL; }
    if (problem     ) { delete   problem     ; problem      = NULL; }
    if (enactor     ) { delete   enactor     ; enactor      = NULL; }
    if (h_sigmas    ) { delete[] h_sigmas    ; h_sigmas     = NULL; }
    if (h_labels    ) { delete[] h_labels    ; h_labels     = NULL; }
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] graph_o  GRGraph type output
 * @param[in]  graph_i  GRGraph type input graph
 * @param[in]  config   Specific configurations
 * @param[in]  data_t   Data type configurations
 * @param[in]  context  ModernGPU context
 */
void dispatch_bc(
    GRGraph*        graph_o,
    const GRGraph*  graph_i,
    const GRSetup   config,
    const GRTypes   data_t,
    ContextPtr*     context,
    cudaStream_t*   streams) {
    Test_Parameter* parameter = new Test_Parameter;
    parameter->context  = context;
    parameter->streams  = streams;
    parameter->num_gpus = config.num_devices;
    parameter->gpu_idx  = config.device_list;

    switch (data_t.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_t.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_t.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT: {  // template type = <int, uint, int>
                // not support yet
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

                // determine source vertex to start
                switch (config.source_mode) {
                case randomize: {
                    parameter->src = graphio::RandomNode(csr.nodes);
                    break;
                }
                case largest_degree: {
                    int max_deg = 0;
                    parameter->src = csr.GetNodeWithHighestDegree(max_deg);
                    break;
                }
                case manually: {
                    parameter->src = config.source_vertex;
                    break;
                }
                default: {
                    parameter->src = 0;
                    break;
                }
                }
                printf(" source: %lld\n", (long long) parameter->src);
                RunTests_instrumented<int, float, int>(graph_o, parameter);

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

/*
 * @brief gunrock_bc function
 *
 * @param[out] graph_o output of bc problem
 * @param[in]  graph_i input graph need to process on
 * @param[in]  config  gunrock primitive specific configurations
 * @param[in]  data_t  gunrock data_t struct
 */
void gunrock_bc(
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

    dispatch_bc(graph_o, graph_i, config, data_t, context, streams);
}

/*
 * @brief Simple interface take in CSR arrays as input
 * @param[out] bfs_label   Return BC node centrality per nodes
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse
 */
void bc(
    float*     bc_scores,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  source) {
    printf("-------------------- setting --------------------\n");

    struct GRTypes data_t;            // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;    // integer
    data_t.SIZET_TYPE = SIZET_INT;    // integer
    data_t.VALUE_TYPE = VALUE_FLOAT;  // float BC scores

    struct GRSetup config;            // primitive-specific configures
    int list[] = {0, 1, 2, 3};        // device to run algorithm
    config.num_devices = sizeof(list) / sizeof(list[0]);  // number of devices
    config.device_list = list;        // device list to run algorithm
    config.source_mode = manually;    // manually setting source vertex
    config.source_vertex = source;    // source vertex to start
    config.max_queue_sizing = 1.0f;   // maximum queue sizing factor

    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    printf(" loaded %d nodes and %d edges\n", num_nodes, num_edges);

    printf("-------------------- running --------------------\n");
    gunrock_bc(graph_o, graph_i, config, data_t);
    memcpy(bc_scores, (float*)graph_o->node_value1, num_nodes * sizeof(float));

    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);

    printf("------------------- completed -------------------\n");
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
