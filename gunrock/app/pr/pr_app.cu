// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file pr_app.cu
 *
 * @brief Gunrock PageRank Implementation
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Page Rank includes
#include <gunrock/app/pr/pr_enactor.cuh>
#include <gunrock/app/pr/pr_problem.cuh>
#include <gunrock/app/pr/pr_functor.cuh>

// Moderngpu include
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::pr;

/**
 * @brief run page rank
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] ggraph_out Pointer to output CSR graph
 * @param[out] node_ids Pointer to output node IDs
 * @param[out] page_rank Pointer to output PageRanks
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source Source ID for personalized PageRank (-1 for general PageRank)
 * @param[in] delta Delta value for computing Page Rank, usually set to .85
 * @param[in] error Error threshold value
 * @param[in] max_iter Max iteration for Page Rank computing
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] context CudaContext for moderngpu to use
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void run_page_rank(
    GunrockGraph   *ggraph_out,
    VertexId       *node_ids,
    Value          *page_rank,
    const Csr<VertexId, Value, SizeT> &graph,
    const VertexId source,
    const Value    delta,
    const Value    error,
    const SizeT    max_iter,
    const int      max_grid_size,
    const int      num_gpus,
    CudaContext&   context) {
    typedef PRProblem <
        VertexId,
        SizeT,
        Value > Problem;

    // Allocate host-side label array for gpu-computed results
    //Value    *h_rank    = (Value*)malloc(sizeof(Value) * graph.nodes);
    //VertexId *h_node_id = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);

    // Allocate Page Rank enactor map
    PREnactor<false> pr_enactor(false);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      false,
                      graph,
                      num_gpus),
                  "PageRank Problem Initialization Failed", __FILE__, __LINE__);

    // Perform PageRank
    GpuTimer gpu_timer;

    util::GRError(csr_problem->Reset(
                      source, delta, error, pr_enactor.GetFrontierType()),
                  "PageRank Problem Data Reset Failed", __FILE__, __LINE__);
    gpu_timer.Start();
    util::GRError(pr_enactor.template Enact<Problem>(
                      context, csr_problem, max_iter, max_grid_size),
                  "PageRank Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();

    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(csr_problem->Extract(page_rank, node_ids),
                  "PageRank Problem Data Extraction Failed",
                  __FILE__, __LINE__);

    // Cleanup
    if (csr_problem) delete csr_problem;
    //if (h_node_id)   free(h_node_id);
    //if (h_rank)      free(h_rank);

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] ggraph_out output of pr problem
 * @param[out] node_ids   output of pr problem
 * @param[out] page_rank  output of pr problem
 * @param[in]  ggraph_in  GunrockGraph type input graph
 * @param[in]  pr_config  pr specific configurations
 * @param[in]  data_type  data type configurations
 * @param[in]  context    moderngpu context
 */
void dispatch_page_rank(
    GunrockGraph          *ggraph_out,
    void                  *node_ids,
    void                  *page_rank,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   pr_config,
    const GunrockDataType data_type,
    CudaContext&          context) {
    switch (data_type.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_type.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_type.VALUE_TYPE) {
            case VALUE_INT: {
                // template type = <int, int, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT: {
                // template type = <int, uint, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {
                // template type = <int, float, int>
                // build input csr format graph
                Csr<int, float, int> csr_graph(false);
                csr_graph.nodes          = ggraph_in->num_nodes;
                csr_graph.edges          = ggraph_in->num_edges;
                csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
                csr_graph.column_indices = (int*)ggraph_in->col_indices;

                // page rank configurations
                float delta         = 0.85f; //!< default delta value
                float error         = 0.01f; //!< error threshold
                int   max_iter      = 20;    //!< maximum number of iterations
                int   max_grid_size = 0;     //!< 0: leave it up to the enactor
                int   num_gpus      = 1;     //!< for multi-gpu enactor to use
                int   src_node      = -1;    //!< source node to start

                // determine source vertex to start sssp
                switch (pr_config.src_mode) {
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
                    src_node = pr_config.src_node;
                    break;
                }
                default: {
                    src_node = -1;
                    break;
                }
                }
                delta    = pr_config.delta;
                error    = pr_config.error;
                max_iter = pr_config.max_iter;

                run_page_rank<int, float, int>(
                    ggraph_out,
                    (int*)node_ids,
                    (float*)page_rank,
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
 * @brief run_page_rank entry
 *
 * @param[out] ggraph_out output of pr problem
 * @param[out] node_ids   output of pr problem
 * @param[out] page_rank  output of pr problem
 * @param[in]  ggraph_in  input graph need to process on
 * @param[in]  pr_config  gunrock primitive specific configurations
 * @param[in]  data_type  gunrock datatype struct
 */
void gunrock_pr_func(
    GunrockGraph          *ggraph_out,
    void                  *node_ids,
    void                  *page_rank,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   pr_config,
    const GunrockDataType data_type) {

    // moderngpu preparations
    int device = 0;
    device = pr_config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // luanch dispatch function
    dispatch_page_rank(
        ggraph_out,
        node_ids,
        page_rank,
        ggraph_in,
        pr_config,
        data_type,
        *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
