// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file mst_app.cu
 *
 * @brief minimum spanning tree (MST) problem implementation
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Primitive-specific includes
#include <gunrock/app/mst/mst_enactor.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

// ModernGPU include
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
 * @param[out] graph_o GunrockGraph type output graph
 * @param[in]  csr Reference to the CSR graph we process on
 * @param[in]  max_grid_size Maximum CTA occupancy
 * @param[in]  num_gpus Number of GPUs
 * @param[in]  context moderngpu context
 */
template<typename VertexId, typename Value, typename SizeT>
void run_mst(
    GunrockGraph   *graph_o,
    const Csr<VertexId, Value, SizeT> &csr,
    const int      max_grid_size,
    const int      num_gpus,
    CudaContext    &context) {
    typedef MSTProblem<VertexId, SizeT, Value, true> Problem;  // preperations
    MSTEnactor<false> enactor(false);                          // enactor map
    VertexId  *h_mst = new VertexId[csr.edges];                // host array
    Problem *problem = new Problem;                            // problem on GPU
    util::GRError(problem->Init(false, csr, num_gpus),
                  "MST Problem Data Initialization Failed", __FILE__, __LINE__);

    util::GRError(problem->Reset(enactor.GetFrontierType()),
                  "MST Problem Data Reset Failed", __FILE__, __LINE__);

    CpuTimer gpu_timer;

    gpu_timer.Start();
    util::GRError(enactor.template Enact<Problem>(
                      context, problem, max_grid_size),
                  "MST Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();
    float elapsed = gpu_timer.ElapsedMillis();

    util::GRError(problem->Extract(h_mst),
                  "MST Problem Data Extraction Failed", __FILE__, __LINE__);

    // output mst results: 0 | 1 mask for all edges
    graph_o->edge_values = (int*)&h_mst[0];

    if (problem) { delete problem; }

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data types
 *
 * @param[out] graph_o  GunrockGraph type output graph
 * @param[in]  graph_i  GunrockGraph type input graph
 * @param[in]  configs  MST-specific configurations
 * @param[in]  datatype data type configurations
 * @param[in]  context  moderngpu context parameter
 */
void dispatch_mst(
    GunrockGraph          *graph_o,
    const GunrockGraph    *graph_i,
    const GunrockConfig   configs,
    const GunrockDataType datatype,
    CudaContext           &context) {
    switch (datatype.VTXID_TYPE) {
    case VTXID_INT: {
        switch (datatype.SIZET_TYPE) {
        case SIZET_INT: {
            switch (datatype.VALUE_TYPE) {
            case VALUE_INT: {  // template type = <int, int, int>
                // create a CSR formatted graph
                Csr<int, int, int> csr(false);
                csr.nodes = graph_i->num_nodes;
                csr.edges = graph_i->num_edges;
                csr.row_offsets    = (int*)graph_i->row_offsets;
                csr.column_indices = (int*)graph_i->col_indices;
                csr.edge_values    = (int*)graph_i->edge_values;
                // configurations if necessary
                int num_gpus      = 1;  // number of GPU(s) to use
                int max_grid_size = 0;  // leave it up tp the enactor
                run_mst<int, int, int>(
                    graph_o, csr, max_grid_size, num_gpus, context);
                // reset for free memory
                csr.row_offsets = NULL;
                csr.column_indices = NULL;
                csr.edge_values = NULL;
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

/**
 * @brief run_mst entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] graph_o  GunrockGraph type output graph
 * @param[in]  graph_i  GunrockGraph type input graph
 * @param[in]  configs  Gunrock primitive-specific configurations
 * @param[in]  datatype data type configurations
 */
void gunrock_mst(
    GunrockGraph          *graph_o,
    const GunrockGraph    *graph_i,
    const GunrockConfig    configs,
    const GunrockDataType  datatype) {
    int device = 0;  // default use GPU 0
    device = configs.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);
    dispatch_mst(graph_o, graph_i, configs, datatype, *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
