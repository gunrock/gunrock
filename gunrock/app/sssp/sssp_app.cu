// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sssp_app.cu
 *
 * @brief single-source shortest path problem implementation
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>

// Moderngpu include
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sssp;

/**
 * @brief run single-source shortest path procedures
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[out] ggraph_out GunrockGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  graph Reference to the CSR graph we process on
 * @param[in]  source Source node where SSSP starts
 * @param[in]  max_grid_size Maximum CTA occupancy
 * @param[in]  queue_sizing Scaling factor used in edge mapping
 * @param[in]  num_gpus Number of GPUs
 * @param[in]  delta_factor user set
 * @param[in]  context moderngpu context
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool MARK_PREDECESSORS >
void run_sssp(
    GunrockGraph   *ggraph_out,
    VertexId       *predecessor,
    const Csr<VertexId, Value, SizeT> &graph,
    const VertexId source,
    const int      max_grid_size,
    const float    queue_sizing,
    const int      num_gpus,
    const int      delta_factor,
    CudaContext& context) {
    // Preparations
    typedef SSSPProblem <
        VertexId,
        SizeT,
        Value,
        MARK_PREDECESSORS > Problem;

    // Allocate host-side label array for gpu-computed results
    unsigned int *h_labels
        = (unsigned int*)malloc(sizeof(unsigned int) * graph.nodes);
    //VertexId     *h_preds  = NULL;

    if (MARK_PREDECESSORS) {
        //h_preds = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
    }

    // Allocate SSSP enactor map
    SSSPEnactor<false> sssp_enactor(false);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      false,
                      graph,
                      num_gpus,
                      delta_factor),
                  "Problem SSSP Initialization Failed", __FILE__, __LINE__);

    // Perform SSSP
    CpuTimer gpu_timer;

    util::GRError(csr_problem->Reset(
                      source, sssp_enactor.GetFrontierType(), queue_sizing),
                  "SSSP Problem Data Reset Failed", __FILE__, __LINE__);
    gpu_timer.Start();
    util::GRError(sssp_enactor.template Enact<Problem>(
                      context, csr_problem, source,
                      queue_sizing, max_grid_size),
                  "SSSP Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();
    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(csr_problem->Extract(h_labels, predecessor),
                  "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    // copy label_values per node to GunrockGraph output
    ggraph_out->node_values = (unsigned int*)&h_labels[0];

    if (csr_problem) delete csr_problem;
    //if (h_labels)    free(h_labels);
    //if (h_preds)     free(h_preds);

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] ggraph_out  GunrockGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  ggraph_in   GunrockGraph type input graph
 * @param[in]  sssp_config sssp specific configurations
 * @param[in]  data_type   sssp data_type configurations
 * @param[in]  context     moderngpu context
 */
void dispatch_sssp(
    GunrockGraph          *ggraph_out,
    void                  *predecessor,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   sssp_config,
    const GunrockDataType data_type,
    CudaContext&          context) {
    switch (data_type.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_type.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_type.VALUE_TYPE) {
            case VALUE_INT: {
                // template type = <int, int, int>
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT: {
                // template type = <int, uint, int>
                // build input csr format graph
                Csr<int, unsigned int, int> csr_graph(false);
                csr_graph.nodes          = ggraph_in->num_nodes;
                csr_graph.edges          = ggraph_in->num_edges;
                csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
                csr_graph.column_indices = (int*)ggraph_in->col_indices;
                csr_graph.edge_values    = (unsigned int*)ggraph_in->edge_values;

                // sssp configurations
                bool  mark_pred        = false;
                int   src_node         = 0; //!< use whatever the specified graph-type's default is
                int   num_gpus         = 1; //!< number of GPUs for multi-gpu enactor to use
                int   delta_factor     = 1; //!< default delta_factor = 1
                int   max_grid_size    = 0; //!< maximum grid size (0: leave it up to the enactor)
                float max_queue_sizing = 1.0; //!< default maximum queue sizing

                // determine source vertex to start sssp
                switch (sssp_config.src_mode) {
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
                    src_node = sssp_config.src_node;
                    break;
                }
                default: {
                    src_node = 0;
                    break;
                }
                }
                mark_pred        = sssp_config.mark_pred;
                delta_factor     = sssp_config.delta_factor;
                max_queue_sizing = sssp_config.queue_size;

                switch (mark_pred) {
                case true: {
                    run_sssp<int, unsigned int, int, true>(
                        ggraph_out,
                        (int*)predecessor,
                        csr_graph,
                        src_node,
                        max_grid_size,
                        max_queue_sizing,
                        num_gpus,
                        delta_factor,
                        context);
                    break;
                }
                case false: {
                    run_sssp<int, unsigned int, int, false>(
                        ggraph_out,
                        (int*)predecessor,
                        csr_graph,
                        src_node,
                        max_grid_size,
                        max_queue_sizing,
                        num_gpus,
                        delta_factor,
                        context);
                    break;
                }
                }
                // reset for free memory
                csr_graph.row_offsets    = NULL;
                csr_graph.column_indices = NULL;
                csr_graph.edge_values    = NULL;
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
 * @param[out] ggraph_out  GunrockGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  ggraph_in   GunrockGraph type input graph
 * @param[in]  sssp_config gunrock primitive specific configurations
 * @param[in]  data_type   data_type configurations
 */
void gunrock_sssp_func(
    GunrockGraph          *ggraph_out,
    void                  *predecessor,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   sssp_config,
    const GunrockDataType data_type) {

    // moderngpu preparations
    int device = 0;
    device = sssp_config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // lunch dispatch function
    dispatch_sssp(
        ggraph_out,
        predecessor,
        ggraph_in,
        sssp_config,
        data_type,
        *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
