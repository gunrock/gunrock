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

/**
 * @brief run single-source shortest path procedures
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[out] graph_o GRGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  graph Reference to the CSR graph we process on
 * @param[in]  source Source node where SSSP starts
 * @param[in]  max_grid_size Maximum CTA occupancy
 * @param[in]  queue_sizing Scaling factor used in edge mapping
 * @param[in]  num_gpus Number of GPUs
 * @param[in]  delta_factor user set
 * @param[in]  context moderngpu context
 */
template<typename VertexId, typename Value, typename SizeT, 
         bool MARK_PREDECESSORS>
void run_sssp(
    GRGraph        *graph_o,
    VertexId       *predecessor,
    const Csr<VertexId, Value, SizeT> &csr,
    const VertexId src,
    const int      max_grid_size,
    const float    queue_sizing,
    const int      num_gpus,
    const int      delta_factor,
    CudaContext    &context) {
    typedef SSSPProblem<VertexId, SizeT, Value, MARK_PREDECESSORS> Problem;
    // Allocate host-side label array for gpu-computed results
    Value *h_labels = (Value*)malloc(sizeof(Value) * csr.nodes);
    //VertexId     *h_preds  = NULL;

    if (MARK_PREDECESSORS) {
        //h_preds = (VertexId*)malloc(sizeof(VertexId) * csr.nodes);
    }

    SSSPEnactor<false> enactor(false);  // enactor map
    Problem *problem = new Problem;
    util::GRError(problem->Init(false, csr, num_gpus, delta_factor),
                  "SSSP Problem Initialization Failed", __FILE__, __LINE__);

    util::GRError(problem->Reset(src, enactor.GetFrontierType(), queue_sizing),
                  "SSSP Problem Data Reset Failed", __FILE__, __LINE__);

    util::GRError(enactor.template Enact<Problem>(
                      context, problem, src, queue_sizing, max_grid_size),
                  "SSSP Problem Enact Failed", __FILE__, __LINE__);

    util::GRError(problem->Extract(h_labels, predecessor),
                  "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    // copy label_values per node to GRGraph output
    graph_o->node_values = (Value*)&h_labels[0];

    if (problem) { delete problem; }
    cudaDeviceSynchronize();
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
void dispatch_sssp(
    GRGraph       *graph_o,
    void          *predecessor,
    const GRGraph *graph_i,
    const GRSetup config,
    const GRTypes data_t,
    CudaContext   &context) {
    switch (data_t.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_t.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_t.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                Csr<int, int, int> csr_graph(false);
                csr_graph.nodes          = graph_i->num_nodes;
                csr_graph.edges          = graph_i->num_edges;
                csr_graph.row_offsets    = (int*)graph_i->row_offsets;
                csr_graph.column_indices = (int*)graph_i->col_indices;
                csr_graph.edge_values    = (int*)graph_i->edge_values;

                // sssp configurations
                bool  mark_pred        =   0;  // whether to mark predecessors
                int   src_node         =   0;  // source vertex to start
                int   num_gpus         =   1;  // number of GPUs
                int   delta_factor     =   1;  // default delta_factor = 1
                int   max_grid_size    =   0;  // leave it up to the enactor
                float max_queue_sizing = 1.0;  // default maximum queue sizing

                // determine source vertex to start sssp
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
                delta_factor     = config.delta_factor;
                max_queue_sizing = config.queue_size;

                switch (mark_pred) {
                case true: {
                    run_sssp<int, int, int, true>(
                        graph_o,
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
                    run_sssp<int, int, int, false>(
                        graph_o,
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
            case VALUE_UINT: {  // template type = <int, uint, int>
                // build input csr format graph
                Csr<int, unsigned int, int> csr_graph(false);
                csr_graph.nodes          = graph_i->num_nodes;
                csr_graph.edges          = graph_i->num_edges;
                csr_graph.row_offsets    = (int*)graph_i->row_offsets;
                csr_graph.column_indices = (int*)graph_i->col_indices;
                csr_graph.edge_values    = (unsigned int*)graph_i->edge_values;

                // sssp configurations
                bool  mark_pred        =   0;  // whether to mark predecessors
                int   src_node         =   0;  // source vertex to start
                int   num_gpus         =   1;  // number of GPUs
                int   delta_factor     =   1;  // default delta_factor = 1
                int   max_grid_size    =   0;  // leave it up to the enactor
                float max_queue_sizing = 1.0;  // default maximum queue sizing

                // determine source vertex to start sssp
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
                delta_factor     = config.delta_factor;
                max_queue_sizing = config.queue_size;

                switch (mark_pred) {
                case true: {
                    run_sssp<int, unsigned int, int, true>(
                        graph_o,
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
                        graph_o,
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
 * @param[out] graph_o     GRGraph type output
 * @param[out] predecessor Return predeessor if mark_pred = true
 * @param[in]  graph_i     GRGraph type input graph
 * @param[in]  config      Primitive specific configurations
 * @param[in]  data_t      Data type configurations
 */
void gunrock_sssp(
    GRGraph       *graph_o,
    void          *predecessor,
    const GRGraph *graph_i,
    const GRSetup config,
    const GRTypes data_t) {
    unsigned int device = 0;
    device = config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);
    dispatch_sssp(graph_o, predecessor, graph_i, config, data_t, *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
