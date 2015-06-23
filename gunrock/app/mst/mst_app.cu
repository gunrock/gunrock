// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file mst_app.cu
 *
 * @brief minimum spanning tree (MST) application
 */

#include <gunrock/gunrock.h>

// graph construction utilities
#include <gunrock/graphio/market.cuh>

// primitive-specific includes
#include <gunrock/app/mst/mst_enactor.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::mst;

/**
 * @brief run minimum spanning tree
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] graph_o   GRGraph type output graph
 * @param[in]  csr       Reference to the CSR graph we process on
 * @param[in]  max_grid  Maximum CTA occupancy
 * @param[in]  num_gpus  Number of GPUs
 * @param[in]  context   Modern GPU context
 */
template<typename VertexId, typename Value, typename SizeT>
void run_mst(
    GRGraph *graph_o,
    const Csr<VertexId, Value, SizeT> &csr,
    const int    max_grid,
    const int    num_gpus,
    CudaContext  &context) {
    typedef MSTProblem<VertexId, SizeT, Value, true> Problem;  // preparations
    MSTEnactor<false> enactor(false);                          // enactor map
    VertexId *h_mst  = new VertexId[csr.edges];                // results array
    Problem *problem = new Problem;                            // problem on GPU

    util::GRError(problem->Init(false, csr, num_gpus),
                  "MST Data Initialization Failed", __FILE__, __LINE__);

    util::GRError(problem->Reset(enactor.GetFrontierType()),
                  "MST Data Reset Failed", __FILE__, __LINE__);

    util::GRError(enactor.template Enact<Problem>(context, problem, max_grid),
                  "MST Enact Failed", __FILE__, __LINE__);

    util::GRError(problem->Extract(h_mst),
                  "MST Data Extraction Failed", __FILE__, __LINE__);

    graph_o->edge_values = (int*)&h_mst[0];  // output: 0|1 mask for all edges

    if (problem) { delete problem; }

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data types
 *
 * @param[out] graph_o  GRGraph type output graph
 * @param[in]  graph_i  GRGraph type input graph
 * @param[in]  config   MST-specific configurations
 * @param[in]  data_t   Data type configurations
 * @param[in]  context  Modern GPU context parameter
 */
void dispatch_mst(
    GRGraph          *graph_o,
    const GRGraph    *graph_i,
    const GRSetup   config,
    const GRTypes data_t,
    CudaContext           &context) {
    switch (data_t.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_t.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_t.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                // create a CSR formatted graph
                Csr<int, int, int> csr(false);
                csr.nodes = graph_i->num_nodes;
                csr.edges = graph_i->num_edges;
                csr.row_offsets    = (int*)graph_i->row_offsets;
                csr.column_indices = (int*)graph_i->col_indices;
                csr.edge_values    = (int*)graph_i->edge_values;

                // configurations if necessary
                int num_gpus = 1;  // number of GPU(s) to use
                int max_grid = 0;  // leave it up to the enactor
                run_mst<int, int, int>(
                    graph_o, csr, max_grid, num_gpus, context);

                // reset for free memory
                csr.row_offsets    = NULL;
                csr.column_indices = NULL;
                csr.edge_values    = NULL;
                break;
            }
            case VALUE_UINT: {  // template type = <int, unsigned int, int>
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT: {  // template type = <int, float, int>
                // create a CSR formatted graph
                Csr<int, float, int> csr(false);
                csr.nodes = graph_i->num_nodes;
                csr.edges = graph_i->num_edges;
                csr.row_offsets    = (int*)graph_i->row_offsets;
                csr.column_indices = (int*)graph_i->col_indices;
                csr.edge_values  = (float*)graph_i->edge_values;

                // configurations if necessary
                int num_gpus = 1;  // number of GPU(s) to use
                int max_grid = 0;  // leave it up to the enactor
                run_mst<int, float, int>(
                    graph_o, csr, max_grid, num_gpus, context);

                // reset for free memory
                csr.row_offsets    = NULL;
                csr.column_indices = NULL;
                csr.edge_values    = NULL;
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
 * @brief run_mst entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] graph_o GRGraph type output graph
 * @param[in]  graph_i GRGraph type input graph
 * @param[in]  config  Primitive-specific configurations
 * @param[in]  data_t  Data type configurations
 */
void gunrock_mst(
    GRGraph       *graph_o,
    const GRGraph *graph_i,
    const GRSetup  config,
    const GRTypes  data_t) {
    unsigned int device = 0;
    device = config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);
    dispatch_mst(graph_o, graph_i, config, data_t, *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
