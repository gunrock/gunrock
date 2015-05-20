// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file bc_app.cu
 *
 * @brief Gunrock Betweeness Centrality Implementation
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BC includes
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
 * @param[out] ggraph_out Pointer to the output CSR graph object
 * @param[in] graph Reference to the CSR graph object defined in main driver
 * @param[in] source
 * @param[in] max_grid_size
 * @param[in] num_gpus
 * @param[in] max_queue_sizing
 * @param[in] context Reference to CudaContext used by moderngpu functions
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void run_bc(
    GunrockGraph *ggraph_out,
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId source,
    int      max_grid_size,
    int      num_gpus,
    double   max_queue_sizing,
    CudaContext& context) {
    typedef BCProblem <
        VertexId,
        SizeT,
        Value,
        true, // MARK_PREDECESSORS
        false > Problem; //does not use double buffer

    // Allocate host-side array (for both reference and gpu-computed results)
    Value *h_sigmas     = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value *h_bc_values  = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value *h_ebc_values = (Value*)malloc(sizeof(Value) * graph.edges);

    // Allocate BC enactor map
    BCEnactor<false> bc_enactor(false);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      false,
                      graph,
                      num_gpus),
                  "BC Problem Initialization Failed", __FILE__, __LINE__);

    // Perform BC
    GpuTimer gpu_timer;

    VertexId start_source;
    VertexId end_source;
    if (source == -1) {
        start_source = 0;
        end_source = graph.nodes;
    } else {
        start_source = source;
        end_source = source + 1;
    }

    gpu_timer.Start();
    for (VertexId i = start_source; i < end_source; ++i) {
        util::GRError(csr_problem->Reset(
                          i, bc_enactor.GetFrontierType(), max_queue_sizing),
                      "BC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(bc_enactor.template Enact<Problem>(
                          context, csr_problem, i, max_grid_size),
                      "BC Problem Enact Failed", __FILE__, __LINE__);
    }

    util::MemsetScaleKernel <<< 128, 128>>>(
        csr_problem->data_slices[0]->d_bc_values, (Value)0.5f, (int)graph.nodes);

    gpu_timer.Stop();

    float elapsed = gpu_timer.ElapsedMillis();

    //double avg_duty = 0.0;
    //bc_enactor.GetStatistics(avg_duty);

    // Copy out results to Host Device
    util::GRError(csr_problem->Extract(h_sigmas, h_bc_values, h_ebc_values),
                  "BC Problem Data Extraction Failed", __FILE__, __LINE__);

    // copy h_bc_values per node to GunrockGraph output
    ggraph_out->node_values = (float*)&h_bc_values[0];
    // copy h_ebc_values per edge to GunrockGraph output
    ggraph_out->edge_values = (float*)&h_ebc_values[0];

    printf("GPU Betweeness Centrality finished in %lf msec.\n", elapsed);

    // Cleanup
    if (csr_problem) delete csr_problem;
    //if (h_sigmas) free(h_sigmas);
    //if (h_bc_values) free(h_bc_values);

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] ggraph_out GunrockGraph type output
 * @param[in]  ggraph_in  GunrockGraph type input graph
 * @param[in]  bc_config  bc specific configurations
 * @param[in]  data_type  bc data_type configurations
 * @param[in]  context    moderngpu context
 */
void dispatch_bc(
    GunrockGraph       *ggraph_out,
    const GunrockGraph *ggraph_in,
    GunrockConfig      bc_config,
    GunrockDataType    data_type,
    CudaContext&       context) {
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
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {
                // template type = <int, float, int>
                // build input csr format graph
                Csr<int, float, int> csr_graph(false);
                csr_graph.nodes = ggraph_in->num_nodes;
                csr_graph.edges = ggraph_in->num_edges;
                csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
                csr_graph.column_indices = (int*)ggraph_in->col_indices;

                // bc configurations
                int   src_node         =  -1; //!< Use whatever the specified graph-type's default is
                int   max_grid_size    =   0; //!< maximum grid size (0: leave it up to the enactor)
                int   num_gpus         =   1; //!< Number of GPUs for multi-gpu enactor to use
                float max_queue_sizing = 1.0; //!< Maximum size scaling factor for work queues

                // determine source vertex to start bc
                switch (bc_config.src_mode) {
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
                    src_node = bc_config.src_node;
                    break;
                }
                default: {
                    src_node = 0;
                    break;
                }
                }
                max_queue_sizing = bc_config.queue_size;

                // lunch bc function
                run_bc<int, float, int>(
                    ggraph_out,
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
 * @param[out] ggraph_out output of bc problem
 * @param[in]  ggraph_in  input graph need to process on
 * @param[in]  bc_config  gunrock primitive specific configurations
 * @param[in]  data_type  gunrock datatype struct
 */
void gunrock_bc_func(
    GunrockGraph       *ggraph_out,
    const GunrockGraph *ggraph_in,
    GunrockConfig      bc_config,
    GunrockDataType    data_type) {

    // moderngpu preparations
    int device = 0;
    device = bc_config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // lunch dispatch function
    dispatch_bc(
        ggraph_out,
        ggraph_in,
        bc_config,
        data_type,
        *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
