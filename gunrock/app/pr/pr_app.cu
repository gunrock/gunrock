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
 * @brief run page-rank
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] graph_o Pointer to output CSR graph
 * @param[out] node_ids Pointer to output node IDs
 * @param[out] page_rank Pointer to output PageRanks
 * @param[in] csr Reference to the CSR graph we process on
 * @param[in] source Source ID for personalized PR (-1 for general PageRank)
 * @param[in] delta Delta value for computing PageRank, usually set to 0.85
 * @param[in] error Error threshold value
 * @param[in] max_iter Max iteration for Page Rank computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 */
template<typename VertexId, typename Value, typename SizeT>
 void run_pagerank(
    GRGraph        *graph_o,
    VertexId       *node_ids,
    Value          *pagerank,
    const Csr<VertexId, Value, SizeT> &csr,
    const Value    delta,
    const Value    error,
    const SizeT    max_iter,
    const int      max_grid_size,
    const int      num_gpus,
    CudaContext&   context) {
    typedef PRProblem<VertexId, SizeT, Value> Problem;
    PREnactor<false> enactor(false);  // PageRank enactor map
    Problem *problem = new Problem;   // Allocate problem on GPU

    util::GRError(problem->Init(false, csr, num_gpus),
                  "PR Problem Initialization Failed", __FILE__, __LINE__);

    util::GRError(problem->Reset(0, delta, error, enactor.GetFrontierType()),
                  "PR Problem Data Reset Failed", __FILE__, __LINE__);

    GpuTimer gpu_timer; float elapsed = 0.0f; gpu_timer.Start();  // start

    util::GRError(enactor.template Enact<Problem>(
                      context, problem, max_iter, max_grid_size),
                  "PR Problem Enact Failed", __FILE__, __LINE__);

    gpu_timer.Stop(); elapsed = gpu_timer.ElapsedMillis();  // elapsed time
    printf(" device elapsed time: %.4f ms\n", elapsed);

    util::GRError(problem->Extract(pagerank, node_ids),
                  "PR Problem Extraction Failed", __FILE__, __LINE__);

    if (problem) delete problem;
    cudaDeviceSynchronize();
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
void dispatch_pagerank(
    GRGraph       *graph_o,
    void          *node_ids,
    void          *pagerank,
    const GRGraph *graph_i,
    const GRSetup  config,
    const GRTypes  data_t,
    CudaContext   &context) {
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
                Csr<int, float, int> csr_graph(false);
                csr_graph.nodes          = graph_i->num_nodes;
                csr_graph.edges          = graph_i->num_edges;
                csr_graph.row_offsets    = (int*)graph_i->row_offsets;
                csr_graph.column_indices = (int*)graph_i->col_indices;

                // pagerank configurations
                float delta         = 0.85f;  // default delta value
                float error         = 0.01f;  // error threshold
                int   max_iter      = 20;     // maximum number of iterations
                int   max_grid_size = 0;      // 0: leave it up to the enactor
                int   num_gpus      = 1;      // for multi-gpu enactor to use

                delta    = config.delta;
                error    = config.error;
                max_iter = config.max_iter;

                run_pagerank<int, float, int>(
                    graph_o,
                    (int*)node_ids,
                    (float*)pagerank,
                    csr_graph,
                    delta,
                    error,
                    max_iter,
                    max_grid_size,
                    num_gpus,
                    context);

                // reset for free memory
                csr_graph.row_offsets    = NULL;
                csr_graph.column_indices = NULL;
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
    void          *node_ids,
    void          *pagerank,
    const GRGraph *graph_i,
    const GRSetup  config,
    const GRTypes  data_t) {
    unsigned int device = 0;
    device = config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);
    dispatch_pagerank(
        graph_o, node_ids, pagerank, graph_i, config, data_t, *context);
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
    printf("-------------------- setting --------------------\n");

    struct GRTypes data_t;            // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;    // integer
    data_t.SIZET_TYPE = SIZET_INT;    // integer
    data_t.VALUE_TYPE = VALUE_FLOAT;  // float ranks

    struct GRSetup config;     // primitive-specific configures
    config.device    =     0;  // setting device to run
    config.delta     = 0.85f;  // default delta value
    config.error     = 0.01f;  // default error threshold
    config.max_iter  =    20;  // maximum number of iterations

    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    printf(" loaded %d nodes and %d edges\n", num_nodes, num_edges);

    printf("-------------------- running --------------------\n");
    gunrock_pagerank(graph_o, node_ids, pagerank, graph_i, config, data_t);

    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);

    printf("------------------- completed -------------------\n");
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
