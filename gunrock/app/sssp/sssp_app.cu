// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sssp_app.cu
 *
 * @brief single-source shortest path (SSSP) application
 */

#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// single-source shortest path includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sssp;

struct Test_Parameter : gunrock::app::TestParameter_Base {
  public:
    bool   mark_predecessors;
    int    delta_factor;
    double max_queue_sizing1;

    Test_Parameter() {
        delta_factor      =    32;
        mark_predecessors = false;
        max_queue_sizing1 =  -1.0;
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
    bool SIZE_CHECK,
    bool MARK_PREDECESSORS >
void runSSSP(GRGraph* output, Test_Parameter *parameter);

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT,
    bool        DEBUG,
    bool        SIZE_CHECK >
void markPredecessorsSSSP(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->mark_predecessors)
        runSSSP<VertexId, Value, SizeT, INSTRUMENT,
                DEBUG, SIZE_CHECK,  true>(output, parameter);
    else
        runSSSP<VertexId, Value, SizeT, INSTRUMENT,
                DEBUG, SIZE_CHECK, false>(output, parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG >
void sizeCheckSSSP(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->size_check)
        markPredecessorsSSSP<VertexId, Value, SizeT, INSTRUMENT,
                             DEBUG,  true>(output, parameter);
    else
        markPredecessorsSSSP<VertexId, Value, SizeT, INSTRUMENT,
                             DEBUG, false>(output, parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT >
void debugSSSP(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->debug)
        sizeCheckSSSP<VertexId, Value, SizeT, INSTRUMENT,
                      true>(output, parameter);
    else
        sizeCheckSSSP<VertexId, Value, SizeT, INSTRUMENT,
                      false>(output, parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT >
void instrumentedSSSP(GRGraph* output, Test_Parameter *parameter) {
    if (parameter->instrumented)
        debugSSSP<VertexId, Value, SizeT,  true>(output, parameter);
    else
        debugSSSP<VertexId, Value, SizeT, false>(output, parameter);
}

/**
 * @brief Run SSSP tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] src Source node where SSSP starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] queue_sizing Scaling factor used in edge mapping
 * @param[in] num_gpus Number of GPUs
 * @param[in] delta_factor Parameter to specify delta in delta-stepping SSSP
 * @param[in] iterations Number of iteration for running the test
 & @param[in] traversal_mode Load-balanced or Dynamic cooperative
 * @param[in] context CudaContext pointer for moderngpu APIs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK,
    bool MARK_PREDECESSORS >
void runSSSP(GRGraph* output, Test_Parameter *parameter) {
    typedef SSSPProblem < VertexId,
            SizeT,
            Value,
            MARK_PREDECESSORS > SsspProblem;

    typedef SSSPEnactor < SsspProblem,
            INSTRUMENT,
            DEBUG,
            SIZE_CHECK > SsspEnactor;

    Csr<VertexId, Value, SizeT>
    *graph = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    VertexId      src                = (VertexId)parameter -> src;
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
    int           delta_factor       = parameter -> delta_factor;
    int           traversal_mode     = parameter -> traversal_mode;
    size_t       *org_size           = new size_t[num_gpus];
    // Allocate host-side label arrays
    Value    *h_labels = new Value[graph->nodes];
    VertexId *h_preds  = MARK_PREDECESSORS ? new VertexId[graph->nodes] : NULL;

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]), &dummy);
    }

    SsspEnactor* enactor = new SsspEnactor(num_gpus, gpu_idx);  // enactor map
    SsspProblem* problem = new SsspProblem;  // Allocate problem on GPU

    util::GRError(
        problem->Init(
            g_stream_from_host,
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
        "Problem SSSP Initialization Failed", __FILE__, __LINE__);

    util::GRError(
        enactor->Init (context, problem, max_grid_size, traversal_mode),
        "SSSP Enactor init failed", __FILE__, __LINE__);

    // Perform SSSP
    CpuTimer cpu_timer;

    util::GRError(
        problem->Reset(src, enactor->GetFrontierType(), max_queue_sizing),
        "SSSP Problem Data Reset Failed", __FILE__, __LINE__);
    util::GRError(
        enactor->Reset(), "SSSP Enactor Reset failed", __FILE__, __LINE__);

    printf("__________________________\n"); fflush(stdout);
    cpu_timer.Start();
    util::GRError(
        enactor->Enact(src, traversal_mode),
        "SSSP Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();
    printf("--------------------------\n"); fflush(stdout);
    float elapsed = cpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(
        problem->Extract(h_labels, h_preds),
        "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    output->node_value1 = (Value*)&h_labels[0];
    if (MARK_PREDECESSORS) output->node_value2 = (VertexId*)&h_preds[0];

    printf(" GPU Single-Source Shortest Path finished in %lf msec.\n", elapsed);

    // Clean up
    if (org_size) { delete[] org_size; org_size = NULL; }
    if (enactor ) { delete   enactor ; enactor  = NULL; }
    if (problem ) { delete   problem ; problem  = NULL; }
    // if (h_labels) { delete[] h_labels; h_labels = NULL; }
    // if (h_preds ) { delete[] h_preds ; h_preds  = NULL; }
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] graph_o     GRGraph type output
 * @param[out] predecessor Return predeessor if mark_pred = true
 * @param[in]  graph_i     GRGraph type input graph
 * @param[in]  config      Primitive-specific configurations
 * @param[in]  data_t      Data type configurations
 * @param[in]  context     ModernGPU context
 */
void dispatchSSSP(
    GRGraph*       graph_o,
    const GRGraph* graph_i,
    const GRSetup  config,
    const GRTypes  data_t,
    ContextPtr*    context,
    cudaStream_t*  streams) {
    Test_Parameter *parameter = new Test_Parameter;
    parameter->context  = context;
    parameter->streams  = streams;
    parameter->num_gpus = config.num_devices;
    parameter->gpu_idx  = config.device_list;
    parameter->delta_factor = config.delta_factor;
    parameter->traversal_mode = config.traversal_mode;
    parameter->mark_predecessors  = config.mark_predecessors;

    switch (data_t.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_t.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_t.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                Csr<int, int, int> csr(false);
                csr.nodes = graph_i->num_nodes;
                csr.edges = graph_i->num_edges;
                csr.row_offsets    = (int*)graph_i->row_offsets;
                csr.column_indices = (int*)graph_i->col_indices;
                csr.edge_values    = (int*)graph_i->edge_values;
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
                instrumentedSSSP<int, int, int>(graph_o, parameter);

                // reset for free memory
                csr.row_offsets    = NULL;
                csr.column_indices = NULL;
                csr.edge_values    = NULL;
                break;
            }
            case VALUE_UINT: {  // template type = <int, uint, int>
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {
                // template type = <int, float, int>
                // not support yet
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
}

/**
 * @brief run_sssp entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] graph_o     GRGraph type output
 * @param[in]  graph_i     GRGraph type input graph
 * @param[in]  config      Primitive specific configurations
 * @param[in]  data_t      Data type configurations
 */
void gunrock_sssp(
    GRGraph*       graph_o,
    const GRGraph* graph_i,
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

    dispatchSSSP(graph_o, graph_i, config, data_t, context, streams);
}

/*
 * @brief Simple interface take in CSR arrays as input
 * @param[out] distances   Return shortest distance to source per nodes
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse
 */
void sssp(
    unsigned int*       distances,
    const int           num_nodes,
    const int           num_edges,
    const int*          row_offsets,
    const int*          col_indices,
    const unsigned int* edge_values,
    const int           source) {
    struct GRTypes data_t;           // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;   // integer
    data_t.SIZET_TYPE = SIZET_INT;   // integer
    data_t.VALUE_TYPE = VALUE_INT;  // unsigned integer

    struct GRSetup config;                // primitive-specific configures
    int list[] = {0, 1, 2, 3};            // device to run algorithm
    config.num_devices = sizeof(list) / sizeof(list[0]);  // number of devices
    config.device_list       = list;      // device list to run algorithm
    config.source_mode       = manually;  // manually setting source vertex
    config.source_vertex     = source;    // source vertex to start
    config.delta_factor      =    32;     // delta factor for SSSP
    config.mark_predecessors = false;     // do not mark predecessors
    config.traversal_mode    =     0;     // 0 for Load balanced partition
    config.max_queue_sizing  =  1.0f;     // maximum queue sizing factor

    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];
    graph_i->edge_values = (void*)&edge_values[0];

    printf(" loaded %d nodes and %d edges\n", num_nodes, num_edges);

    gunrock_sssp(graph_o, graph_i, config, data_t);
    memcpy(distances, (int*)graph_o->node_value1, num_nodes * sizeof(int));

    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
