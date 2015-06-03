// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file cc_app.cu
 *
 * @brief connected component implementation.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// CC includes
#include <gunrock/app/cc/cc_enactor.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::cc;

/**
 * @brief Run tests for connected component algorithm
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] ggraph_out Pointer to output CSR graph
 * @param[in] csr_graph Reference to the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy for CC kernels
 * @param[in] num_gpus Number of GPUs
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void run_cc(
    GunrockGraph *ggraph_out,
    unsigned int *components,
    const Csr<VertexId, Value, SizeT> &csr_graph,
    const int    max_grid_size,
    const int    num_gpus) {

    // Define CCProblem
    typedef CCProblem <
        VertexId,
        SizeT,
        Value,
        true > Problem; //use double buffer

    // Allocate host-side label array for gpu-computed results
    VertexId *h_component_ids
        = (VertexId*)malloc(sizeof(VertexId) * csr_graph.nodes);

    // Allocate CC enactor map
    CCEnactor<false> cc_enactor(false);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
                      false,
                      csr_graph,
                      num_gpus),
                  "CC Problem Initialization Failed", __FILE__, __LINE__);

    // Reset CC Problem Data
    util::GRError(csr_problem->Reset(
                      cc_enactor.GetFrontierType()),
                  "CC Problem Data Reset Failed", __FILE__, __LINE__);

    // Perform Connected Component
    GpuTimer gpu_timer;
    gpu_timer.Start();
    // Lunch CC Enactor
    util::GRError(cc_enactor.template Enact<Problem>(
                      csr_problem, max_grid_size),
                  "CC Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();
    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results back to Host Device
    util::GRError(csr_problem->Extract(h_component_ids),
                  "CC Problem Data Extraction Failed", __FILE__, __LINE__);

    // Compute number of components in graph
    unsigned int temp = csr_problem->num_components;
    *components = temp;

    // copy component_id per node to GunrockGraph struct
    ggraph_out->node_values = (int*)&h_component_ids[0];

    printf("GPU Connected Component finished in %lf msec.\n", elapsed);

    // Cleanup
    if (csr_problem)  delete csr_problem;

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] ggraph_out GunrockGraph type output
 * @param[in]  ggraph_in  GunrockGraph type input graph
 * @param[in]  cc_config  cc specific configurations
 * @param[in]  data_type  data type configurations
 */
void dispatch_cc(
    GunrockGraph          *ggraph_out,
    unsigned int          *components,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   cc_config,
    const GunrockDataType data_type) {
    switch (data_type.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_type.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_type.VALUE_TYPE) {
            case VALUE_INT: {
                // template type = <int, int, int>
                // build input csr format graph
                Csr<int, int, int> csr_graph(false);
                csr_graph.nodes = ggraph_in->num_nodes;
                csr_graph.edges = ggraph_in->num_edges;
                csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
                csr_graph.column_indices = (int*)ggraph_in->col_indices;

                int max_grid_size = 0; //!< 0: leave it up to the enactor
                int num_gpus      = 1; //!< number of GPUs

                // lunch cc dispatch function
                run_cc<int, int, int>(
                    ggraph_out,
                    (unsigned int*)components,
                    csr_graph,
                    max_grid_size,
                    num_gpus);

                // reset for free memory
                csr_graph.row_offsets    = NULL;
                csr_graph.column_indices = NULL;
                break;
            }
            case VALUE_UINT: {
                // template type = <int, uint, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {
                // template type = <int, float, int>
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
 * @brief gunrock_cc function
 *
 * @param[out] ggraph_out output subgraph of cc problem
 * @param[in]  ggraph_in  input graph need to process on
 * @param[in]  cc_configs primitive specific configurations
 * @param[in]  data_type  gunrock data_type struct
 */
void gunrock_cc_func(
    GunrockGraph          *ggraph_out,
    unsigned int          *components,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   cc_configs,
    const GunrockDataType data_type) {

    // lunch dispatch function
    dispatch_cc(ggraph_out, components, ggraph_in, cc_configs, data_type);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
