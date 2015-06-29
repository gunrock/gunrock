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
 * @brief Run BFS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[out] graph_o Pointer to the output CSR graph
 * @param[in] graph_i Reference to the CSR graph we process on
 * @param[in] src Source node where BFS starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] max_queue_sizing Scaling factor used in edge mapping
 * @param[in] context Reference to CudaContext used by moderngpu functions
 *
 */
template<typename VertexId, typename Value, typename SizeT,
         bool MARK_PREDECESSORS, bool ENABLE_IDEMPOTENCE>
void run_bfs(
    GRGraph*       graph_o,
    const Csr<VertexId, Value, SizeT>& csr,
    const VertexId src,
    const int      num_gpus,
    const double   max_queue_sizing,
    CudaContext&   context) {
    typedef BFSProblem<VertexId, SizeT, Value, MARK_PREDECESSORS,
        ENABLE_IDEMPOTENCE, (MARK_PREDECESSORS && ENABLE_IDEMPOTENCE)> Problem;
    // Allocate host-side label array for GPU-computed results
    VertexId *h_labels = (VertexId*)malloc(sizeof(VertexId) * csr.nodes);
    VertexId *h_preds = NULL;
    if (MARK_PREDECESSORS) {
        // h_preds = (VertexId*)malloc(sizeof(VertexId) * csr.nodes);
    }

    BFSEnactor<false> enactor(false);  // Allocate BFS enactor map
    Problem *problem = new Problem;    // Allocate problem on GPU

    util::GRError(problem->Init(false, csr, num_gpus),
                  "BFS Problem Initialization Failed", __FILE__, __LINE__);

    util::GRError(problem->Reset(
                      src, enactor.GetFrontierType(), max_queue_sizing),
                  "BFS Problem Data Reset Failed", __FILE__, __LINE__);

    GpuTimer gpu_timer; float elapsed = 0.0f; gpu_timer.Start();  // start

    util::GRError(enactor.template Enact<Problem>(context, problem, src),
                  "BFS Problem Enact Failed", __FILE__, __LINE__);

    gpu_timer.Stop(); elapsed = gpu_timer.ElapsedMillis();  // elapsed time
    printf(" device elapsed time: %.4f ms\n", elapsed);

    util::GRError(problem->Extract(h_labels, h_preds),
                  "BFS Problem Data Extraction Failed", __FILE__, __LINE__);

    graph_o->node_values = (int*)&h_labels[0];  // label per node to graph_o

    if (problem) { delete problem; }
    if (h_preds) {  free(h_preds); }
    cudaDeviceSynchronize();
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
void dispatch_bfs(
    GRGraph*       graph_o,
    const GRGraph* graph_i,
    const GRSetup  config,
    const GRTypes  data_t,
    CudaContext&   context) {
    switch (data_t.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_t.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_t.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                // build input csr format graph
                Csr<int, int, int> csr_graph(false);
                csr_graph.nodes = graph_i->num_nodes;
                csr_graph.edges = graph_i->num_edges;
                csr_graph.row_offsets    = (int*)graph_i->row_offsets;
                csr_graph.column_indices = (int*)graph_i->col_indices;

                // default configurations
                int   src_node      = 0;  // default source vertex to start
                int   num_gpus      = 1;  // number of GPUs for multi-GPU
                bool  mark_pred     = 0;  // whether to mark predecessor or not
                bool  idempotence   = 0;  // whether or not enable idempotent
                float max_queue_sizing = 1.0f;  // maximum size scaling factor

                // determine source vertex to start
                switch (config.src_mode) {
                case randomize: {
                    src_node = graphio::RandomNode(csr_graph.nodes);
                    break;
                }
                case largest_degree: {
                    int max_deg = 0;
                    src_node = csr_graph.GetNodeWithHighestDegree(max_deg);
                    break;
                }
                case manually: {
                    src_node = config.src_node;
                    break;
                }
                default: {
                    src_node = 0;
                    break;
                }
                }
                mark_pred        = config.mark_pred;
                idempotence      = config.idempotence;
                max_queue_sizing = config.queue_size;

                if (mark_pred) {
                    if (idempotence) {
                        run_bfs<int, int, int, true, true>(
                            graph_o,
                            csr_graph,
                            src_node,
                            num_gpus,
                            max_queue_sizing,
                            context);
                    } else {
                        run_bfs<int, int, int, true, false>(
                            graph_o,
                            csr_graph,
                            src_node,
                            num_gpus,
                            max_queue_sizing,
                            context);
                    }
                } else {
                    if (idempotence) {
                        run_bfs<int, int, int, false, true>(
                            graph_o,
                            csr_graph,
                            src_node,
                            num_gpus,
                            max_queue_sizing,
                            context);
                    } else {
                        run_bfs<int, int, int, false, false>(
                            graph_o,
                            csr_graph,
                            src_node,
                            num_gpus,
                            max_queue_sizing,
                            context);
                    }
                }
                // reset for free memory
                csr_graph.row_offsets    = NULL;
                csr_graph.column_indices = NULL;
                break;
            }
            case VALUE_UINT: {  // template type = <int, uint, int>
                // not yet support
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {  // template type = <int, float, int>
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
}

/*
 * @brief gunrock_bfs function
 *
 * @param[out] graph_o output subgraph of the problem
 * @param[in]  graph_i input graph need to process on
 * @param[in]  config  gunrock primitive specific configurations
 * @param[in]  data_t  gunrock data_t struct
 */
void gunrock_bfs(
    GRGraph*       graph_o,
    const GRGraph* graph_i,
    const GRSetup  config,
    const GRTypes  data_t) {
    unsigned int device = 0;
    device = config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);
    dispatch_bfs(graph_o, graph_i, config, data_t, *context);
}

/*
 * @brief Simple interface take in CSR arrays as input
 * @param[out] bfs_label   Return BFS labels per nodes
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse
 */
void bfs(
    int*       bfs_label,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices,
    const int  source) {
    printf("-------------------- setting --------------------\n");

    struct GRTypes data_t;          // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;  // integer
    data_t.SIZET_TYPE = SIZET_INT;  // integer
    data_t.VALUE_TYPE = VALUE_INT;  // integer

    struct GRSetup config;          // primitive-specific configures
    config.device      =      0;    // setting device to run
    config.src_node    = source;    // source vertex to begin
    config.mark_pred   =  false;    // do not mark predecessors
    config.idempotence =  false;    // whether enable idempotent
    config.queue_size  =   1.0f;    // maximum queue size factor

    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    printf(" loaded %d nodes and %d edges\n", num_nodes, num_edges);

    printf("-------------------- running --------------------\n");
    gunrock_bfs(graph_o, graph_i, config, data_t);
    memcpy(bfs_label, (int*)graph_o->node_values, num_nodes * sizeof(int));

    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);

    printf("------------------- completed -------------------\n");
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
