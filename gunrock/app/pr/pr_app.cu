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
    Value          *page_rank,
    const Csr<VertexId, Value, SizeT> &csr,
    const VertexId source,
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

    util::GRError(problem->Reset(
                      source, delta, error, enactor.GetFrontierType()),
                  "PR Problem Data Reset Failed", __FILE__, __LINE__);

    util::GRError(enactor.template Enact<Problem>(
                      context, problem, max_iter, max_grid_size),
                  "PR Problem Enact Failed", __FILE__, __LINE__);

    util::GRError(problem->Extract(page_rank, node_ids),
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

                // page-rank configurations
                float delta         = 0.85f;  // default delta value
                float error         = 0.01f;  // error threshold
                int   max_iter      = 20;     // maximum number of iterations
                int   max_grid_size = 0;      // 0: leave it up to the enactor
                int   num_gpus      = 1;      // for multi-gpu enactor to use
                int   src_node      = -1;     // source node to start

                // determine source vertex to start sssp
                switch (config.src_mode) {
                case randomize: {
                    src_node = graphio::RandomNode(csr_graph.nodes);
                    break;
                }
                case largest_degree: {
                    int max_node = 0;
                    src_node = csr_graph.GetNodeWithHighestDegree(max_node);
                    break;
                }
                case manually: {
                    src_node = config.src_node;
                    break;
                }
                default: {
                    src_node = -1;
                    break;
                }
                }
                delta    = config.delta;
                error    = config.error;
                max_iter = config.max_iter;

                run_pagerank<int, float, int>(
                    graph_o,
                    (int*)node_ids,
                    (float*)pagerank,
                    csr_graph,
                    src_node,
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

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
