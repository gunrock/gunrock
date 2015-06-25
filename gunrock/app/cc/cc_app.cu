// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file cc_app.cu
 *
 * @brief connected component (CC) application
 */

#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// connected component includes
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
 * @param[out] graph_o Pointer to output CSR graph
 * @param[in] csr_graph Reference to the CSR graph we process on
 * @param[in] max_grid_size Maximum CTA occupancy for CC kernels
 * @param[in] num_gpus Number of GPUs
 */
template<typename VertexId, typename Value, typename SizeT>
void run_cc(
    GRGraph*      graph_o,
    unsigned int* components,
    const Csr<VertexId, Value, SizeT>& csr,
    const int    max_grid_size,
    const int    num_gpus) {
    typedef CCProblem<VertexId, SizeT, Value, true> Problem; // double buffer

    // Allocate host-side label array for gpu-computed results
    VertexId *h_component_ids
        = (VertexId*)malloc(sizeof(VertexId) * csr.nodes);    
    CCEnactor<false> cc_enactor(false);  // Allocate CC enactor map
    Problem *problem = new Problem;  // Allocate problem on GPU

    util::GRError(problem->Init(false, csr, num_gpus),
                  "CC Problem Initialization Failed", __FILE__, __LINE__);

    util::GRError(problem->Reset(
                      cc_enactor.GetFrontierType()),
                  "CC Problem Data Reset Failed", __FILE__, __LINE__);

    GpuTimer gpu_timer; float elapsed = 0.0f; gpu_timer.Start();  // start timer

    util::GRError(cc_enactor.template Enact<Problem>(
                      problem, max_grid_size),
                  "CC Problem Enact Failed", __FILE__, __LINE__);

    gpu_timer.Stop(); elapsed = gpu_timer.ElapsedMillis();  // calculate elapsed
    printf(" device elapsed time: %.4f ms\n", elapsed);

    util::GRError(problem->Extract(h_component_ids),
                  "CC Problem Data Extraction Failed", __FILE__, __LINE__);

    // Compute number of components in graph
    unsigned int temp = problem->num_components;
    *components = temp;

    // copy component_id per node to GRGraph struct
    graph_o->node_values = (int*)&h_component_ids[0];

    if (problem)  delete problem;
    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] graph_o GRGraph type output
 * @param[in]  graph_i GRGraph type input graph
 * @param[in]  config  cc specific configurations
 * @param[in]  data_t  data type configurations
 */
void dispatch_cc(
    GRGraph*       graph_o,
    unsigned int*  components,
    const GRGraph* graph_i,
    const GRSetup  config,
    const GRTypes  data_t) {
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

                int max_grid_size = 0;  // 0: leave it up to the enactor
                int num_gpus      = 1;  // number of GPUs

                run_cc<int, int, int>(
                    graph_o,
                    (unsigned int*)components,
                    csr_graph,
                    max_grid_size,
                    num_gpus);

                // reset for free memory
                csr_graph.row_offsets    = NULL;
                csr_graph.column_indices = NULL;
                break;
            }
            case VALUE_UINT: {  // template type = <int, uint, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {  // template type = <int, float, int>
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
 * @param[out] graph_o output subgraph of cc problem
 * @param[in]  graph_i input graph need to process on
 * @param[in]  config  primitive specific configurations
 * @param[in]  data_t  gunrock data_t struct
 */
void gunrock_cc(
    GRGraph       *graph_o,
    unsigned int  *components,
    const GRGraph *graph_i,
    const GRSetup  config,
    const GRTypes  data_t) {
    dispatch_cc(graph_o, components, graph_i, config, data_t);
}

/*
 * @brief Simple interface take in CSR arrays as input
 * @param[out] components  Return component ID for each node
 * @param[out] num_comps   Return number of components calculated
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 */
int cc(
    int*       components,
    const int  num_nodes,
    const int  num_edges,
    const int* row_offsets,
    const int* col_indices) {
    printf("-------------------- setting --------------------\n");

    struct GRTypes data_t;          // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;  // integer
    data_t.SIZET_TYPE = SIZET_INT;  // integer
    data_t.VALUE_TYPE = VALUE_INT;  // integer

    struct GRSetup config;  // primitive-specific configures
    config.device = 0;      // setting device to run

    unsigned int num_components = 0;
    struct GRGraph *graph_o = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graph_i = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    graph_i->num_nodes   = num_nodes;
    graph_i->num_edges   = num_edges;
    graph_i->row_offsets = (void*)&row_offsets[0];
    graph_i->col_indices = (void*)&col_indices[0];

    printf(" loaded %d nodes and %d edges\n", num_nodes, num_edges);    
    
    printf("-------------------- running --------------------\n");
    gunrock_cc(graph_o, &num_components, graph_i, config, data_t);
    memcpy(components, (int*)graph_o->node_values, num_nodes * sizeof(int));
    
    if (graph_i) free(graph_i);
    if (graph_o) free(graph_o);

    printf("------------------- completed -------------------\n");
    return num_components;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
