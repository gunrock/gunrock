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
 * @brief Run betweenness centrality tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] graph_o Pointer to the output CSR graph object
 * @param[in] csr Reference to the CSR graph object defined in main driver
 * @param[in] source
 * @param[in] max_grid_size
 * @param[in] num_gpus
 * @param[in] max_queue_sizing
 * @param[in] context Reference to CudaContext used by moderngpu functions
 */
template<typename VertexId, typename Value, typename SizeT>
void run_bc(
    GRGraph        *graph_o,
    const Csr<VertexId, Value, SizeT> &csr,
    const VertexId source,
    const int      max_grid_size,
    const int      num_gpus,
    const double   max_queue_sizing,
    CudaContext    &context) {
    typedef BCProblem<VertexId, SizeT, Value, true, false > Problem;
    // Allocate host-side array (for both reference and gpu-computed results)
    Value *h_sigmas     = (Value*)malloc(sizeof(Value) * csr.nodes);
    Value *h_bc_values  = (Value*)malloc(sizeof(Value) * csr.nodes);
    Value *h_ebc_values = (Value*)malloc(sizeof(Value) * csr.edges);
    BCEnactor<false> enactor(false);  // Allocate BC enactor map
    Problem *problem = new Problem;   // Allocate problem on GPU

    util::GRError(problem->Init(false, csr, num_gpus),
                  "BC Problem Initialization Failed", __FILE__, __LINE__);

    VertexId start_source;
    VertexId end_source;
    if (source == -1) {
        start_source = 0;
        end_source = csr.nodes;
    } else {
        start_source = source;
        end_source = source + 1;
    }

    for (VertexId i = start_source; i < end_source; ++i) {
        util::GRError(problem->Reset(
                          i, enactor.GetFrontierType(), max_queue_sizing),
                      "BC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(enactor.template Enact<Problem>(
                          context, problem, i, max_grid_size),
                      "BC Problem Enact Failed", __FILE__, __LINE__);
    }

    util::MemsetScaleKernel <<< 128, 128>>>(
        problem->data_slices[0]->d_bc_values, (Value)0.5f, (int)csr.nodes);

    util::GRError(problem->Extract(h_sigmas, h_bc_values, h_ebc_values),
                  "BC Problem Data Extraction Failed", __FILE__, __LINE__);

    graph_o->node_values = (float*)&h_bc_values[0];   // h_bc_values per node 
    graph_o->edge_values = (float*)&h_ebc_values[0];  // h_ebc_values per edge

    if (problem) { delete problem; }
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
void dispatch_bc(
    GRGraph       *graph_o,
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
                Csr<int, float, int> csr_graph(false);
                csr_graph.nodes = graph_i->num_nodes;
                csr_graph.edges = graph_i->num_edges;
                csr_graph.row_offsets    = (int*)graph_i->row_offsets;
                csr_graph.column_indices = (int*)graph_i->col_indices;

                // bc configurations
                int   src_node         =  -1;  // default source vertex to start
                int   max_grid_size    =   0;  // leave it up to the enactor
                int   num_gpus         =   1;  // Number of GPUs for multi-gpu
                float max_queue_sizing = 1.0;  // Maximum size scaling factor

                // determine source vertex to start bc
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
                max_queue_sizing = config.queue_size;

                // lunch bc function
                run_bc<int, float, int>(
                    graph_o,
                    csr_graph,
                    src_node,
                    max_grid_size,
                    num_gpus,
                    max_queue_sizing,
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
    unsigned int device = 0;
    device = config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);
    dispatch_bc(graph_o, graph_i, config, data_t, *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
