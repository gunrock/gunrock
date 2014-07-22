// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bc.cu
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


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/
bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf(
        "\ntest_bc <graph type> <graph type args> [--device=<device_index>] "
        "[--instrumented] [--source=<source index>] [--quick] [--v]"
        "[--queue-sizing=<scale factor>] [--ref-file=<reference filename>]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "--device=<device_index>: Set GPU device for running the graph primitive.\n"
        "--undirected: If set then treat the graph as undirected graph.\n"
        "--instrumented: If set then kernels keep track of queue-search_depth\n"
        "and barrier duty (a relative indicator of load imbalance.)\n"
        "--source=<source index>: When source index is -1, compute BC value for each\n"
        "node. Otherwise, debug the delta value for one node\n"
        "--quick: If set will skip the CPU validation code.\n"
        "--queue-sizing Allocates a frontier queue sized at (graph-edges * <scale factor>).\n"
        "Default is 1.0.\n"
        "--v: If set, enable verbose output, keep track of the kernel running.\n"
        "--ref-file: If set, use pre-computed result stored in ref-file to verify.\n"
        );
}

/**
 * @brief Displays the BC result (sigma value and BC value)
 *
 * @param[in] sigmas
 * @param[in] bc_values
 * @param[in] nodes
 */
template<typename Value, typename SizeT>
void DisplaySolution(Value *sigmas, Value *bc_values, SizeT nodes)
{
    if (nodes < 40)
    {
        printf("[");
        for (SizeT i = 0; i < nodes; ++i)
        {
            PrintValue(i);
            printf(":");
            PrintValue(sigmas[i]);
            printf(",");
            PrintValue(bc_values[i]);
            printf(" ");
        }
        printf("]\n");
    }
}

/******************************************************************************
 * BC Testing Routines
 *****************************************************************************/
/**
 * @brief Run betweenness centrality tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph object defined in main driver
 * @param[in] source
 * @param[in] max_grid_size
 * @param[in] num_gpus
 * @param[in] max_queue_sizing
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void run_bc(
    GunrockGraph *ggraph_out,
    const Csr<VertexId, Value, SizeT> &graph,
    VertexId source,
    int      max_grid_size,
    int      num_gpus,
    double   max_queue_sizing,
    CudaContext& context)
{
    typedef BCProblem<
        VertexId,
        SizeT,
        Value,
        true, // MARK_PREDECESSORS
        false> Problem; //does not use double buffer

    // Allocate host-side array (for both reference and gpu-computed results)
    Value *h_sigmas     = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value *h_bc_values  = (Value*)malloc(sizeof(Value) * graph.nodes);
    Value *h_ebc_values = (Value*)malloc(sizeof(Value) * graph.edges);

    // Allocate BC enactor map
    BCEnactor<false> bc_enactor(g_verbose);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
        g_stream_from_host,
        graph,
        num_gpus),
        "BC Problem Initialization Failed", __FILE__, __LINE__);

    double avg_duty = 0.0;

    // Perform BC
    GpuTimer gpu_timer;

    VertexId start_source;
    VertexId end_source;
    if (source == -1)
    {
        start_source = 0;
        end_source = graph.nodes;
    }
    else
    {
        start_source = source;
        end_source = source + 1;
    }

    gpu_timer.Start();
    for (VertexId i = start_source; i < end_source; ++i)
    {
        util::GRError(csr_problem->Reset(
            i, bc_enactor.GetFrontierType(), max_queue_sizing),
            "BC Problem Data Reset Failed", __FILE__, __LINE__);
        util::GRError(bc_enactor.template Enact<Problem>(
            context, csr_problem, i, max_grid_size),
            "BC Problem Enact Failed", __FILE__, __LINE__);
    }

    util::MemsetScaleKernel<<<128, 128>>>(
        csr_problem->data_slices[0]->d_bc_values, (Value)0.5f, (int)graph.nodes);

    gpu_timer.Stop();

    float elapsed = gpu_timer.ElapsedMillis();

    bc_enactor.GetStatistics(avg_duty);

    // Copy out results to Host Device
    util::GRError(csr_problem->Extract(h_sigmas, h_bc_values, h_ebc_values),
        "BC Problem Data Extraction Failed", __FILE__, __LINE__);

    // copy h_bc_values per node to GunrockGraph output
    ggraph_out->node_values = (float*)&h_bc_values[0];
    // copy h_ebc_values per edge to GunrockGraph output
    ggraph_out->edge_values = (float*)&h_ebc_values[0];

    // Display Solution
    DisplaySolution(h_sigmas, h_bc_values, graph.nodes);

    printf("GPU BC finished in %lf msec.\n", elapsed);
    if (avg_duty != 0)
    {
        printf("\n avg CTA duty: %.2f%% \n", avg_duty * 100);
    }

    // Cleanup
    if (csr_problem) delete csr_problem;
    //if (h_sigmas) free(h_sigmas);
    //if (h_bc_values) free(h_bc_values);

    cudaDeviceSynchronize();
}

/*
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void dispatch_bc(
    Csr<VertexId, Value, SizeT> &graph,
    CommandLineArgs &args,
    CudaContext& context)
{

    if (instrumented) {
        RunTests<VertexId, Value, SizeT, true>(
            graph,
            source,
            max_grid_size,
            num_gpus,
            max_queue_sizing,
            context);
    } else {
        RunTests<VertexId, Value, SizeT, false>(
            graph,
            source,
            max_grid_size,
            num_gpus,
            max_queue_sizing,
            context);
    }

}
*/

/*
* @brief gunrock_bc function
*
* @param[out] output of bc problem
* @param[in]  input graph need to process on
* @param[in]  gunrock bc configurations
* @param[in]  gunrock datatype struct
*/
void gunrock_bc(
    GunrockGraph       *ggraph_out,
    const GunrockGraph *ggraph_in,
    GunrockConfig      bc_config,
    GunrockDataType    data_type)
{
    // moderngpu preparations
    int device = 0;
    device = bc_config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // build input csr format graph
    Csr<int, float, int> csr_graph(false);
    csr_graph.nodes = ggraph_in->num_nodes;
    csr_graph.edges = ggraph_in->num_edges;
    csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
    csr_graph.column_indices = (int*)ggraph_in->col_indices;

    // bc configurations
    int   source           =   -1; //!< Use whatever the specified graph-type's default is
    int   max_grid_size    =    0; //!< maximum grid size (0: leave it up to the enactor)
    int   num_gpus         =    1; //!< Number of GPUs for multi-gpu enactor to use
    double max_queue_sizing = 1.0; //!< Maximum size scaling factor for work queues

    source           = bc_config.source;
    max_queue_sizing = bc_config.queue_size;

    // lunch bc dispatch function
    run_bc<int, float, int>(
        ggraph_out,
        csr_graph,
        source,
        max_grid_size,
        num_gpus,
        max_queue_sizing,
        *context);

    // reset for free memory
    csr_graph.row_offsets    = NULL;
    csr_graph.column_indices = NULL;
    csr_graph.row_offsets    = NULL;
    csr_graph.column_indices = NULL;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: