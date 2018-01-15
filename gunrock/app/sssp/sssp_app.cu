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

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>

// single-source shortest path includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_test.cuh>

namespace gunrock {
namespace app {
namespace sssp {

cudaError_t UseParameters(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    //GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(UseParameters_app    (parameters));
    GUARD_CU(UseParameters_problem(parameters));
    GUARD_CU(UseParameters_enactor(parameters));

    GUARD_CU(parameters.Use<std::string>(
        "src",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        "0",
        "<Vertex-ID|random|largestdegree> The source vertices\n"
        "\tIf random, randomly select non-zero degree vertices;\n"
        "\tIf largestdegree, select vertices with largest degrees",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "src-seed",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        util::PreDefinedValues<int>::InvalidValue,
        "seed to generate random sources",
        __FILE__, __LINE__));

    return retval;
}

/**
 * @brief Run SSSP tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <typename GraphT>
cudaError_t RunTests(
    util::Parameters &parameters,
    GraphT           &graph,
    util::Location target = util::DEVICE)
{
    cudaError_t retval = cudaSuccess;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef Problem<GraphT  > ProblemT;
    typedef Enactor<ProblemT> EnactorT;

    // parse configurations from parameters
    bool quiet_mode = parameters.Get<bool>("quiet");
    bool quick_mode = parameters.Get<bool>("quick");
    bool mark_pred  = parameters.Get<bool>("mark-pred");
    int  num_runs   = parameters.Get<int >("num-runs");
    std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
    int  num_srcs   = srcs   .size();

    util::CpuTimer    cpu_timer;
    cpu_timer.Start();
    //Info<VertexT, SizeT, ValueT> *info = new Info<VertexT, SizeT, ValueT>;
    //info->Init("SSSP", parameters, graph);  // initialize Info structure
    //info->info["load_time"] = cpu_timer2.ElapsedMillis();

    // Allocate host-side array (for both reference and GPU-computed results)
    ValueT  *h_distances = new ValueT[graph.nodes];
    VertexT *h_preds = (mark_pred ) ? new VertexT[graph.nodes] : NULL;

    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem(parameters);
    EnactorT enactor;
    GUARD_CU(problem.Init(graph  , target));
    GUARD_CU(enactor.Init(problem, target));
    cpu_timer.Stop();
    //info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // perform SSSP
    //double total_elapsed  = 0.0;
    double single_elapsed = 0.0;
    //double max_elapsed    = 0.0;
    //double min_elapsed    = 1e10;
    //json_spirit::mArray process_times;

    util::PrintMsg("Using advance mode "
        + parameters.Get<std::string>("advance-mode"), !quiet_mode);
    util::PrintMsg("Using filter mode "
        + parameters.Get<std::string>("filter-mode"), !quiet_mode);

    VertexT src;
    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        src = srcs[run_num % num_srcs];
        GUARD_CU(problem.Reset(src, target));
        GUARD_CU(enactor.Reset(src, target));
        util::PrintMsg("__________________________", !quiet_mode);

        cpu_timer.Start();
        GUARD_CU(enactor.Enact(src));
        cpu_timer.Stop();
        single_elapsed = cpu_timer.ElapsedMillis();
        //total_elapsed += single_elapsed;
        //process_times.push_back(single_elapsed);
        //if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
        //if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;
        util::PrintMsg("--------------------------\nRun "
            + std::to_string(run_num) + " elapsed: "
            + std::to_string(single_elapsed) + " ms, src = "
            + std::to_string(src) + ", #iterations = "
            + std::to_string(enactor.enactor_slices[0]
                .enactor_stats.iteration), !quiet_mode);
    }
    //total_elapsed /= num_runs;
    //info -> info["process_times"] = process_times;
    //info -> info["min_process_time"] = min_elapsed;
    //info -> info["max_process_time"] = max_elapsed;

    cpu_timer.Start();
    // Copy out results
    GUARD_CU(problem.Extract(h_distances, h_preds));
    SizeT num_errors = app::sssp::Validate_Results(
        parameters, graph, src, h_distances, h_preds);

    //info->ComputeTraversalStats(  // compute running statistics
    //    enactor.enactor_stats.GetPointer(), total_elapsed, h_distances);

    if (!quiet_mode)
    {
        //Display_Memory_Usage(num_gpus, gpu_idx, org_size, problem);
        #ifdef ENABLE_PERFORMANCE_PROFILING
            //Display_Performance_Profiling(enactor);
        #endif
    }

    // Clean up
    util::PrintMsg("1");
    GUARD_CU(enactor.Release(target));
    util::PrintMsg("2");
    GUARD_CU(problem.Release(target));
    delete[] h_distances  ; h_distances   = NULL;
    util::PrintMsg("3");
    delete[] h_preds      ; h_preds       = NULL;
    util::PrintMsg("4");
    cpu_timer.Stop();
    //info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    //info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!parameters.Get<bool>("quiet"))
    {
        //info->DisplayStats();  // display collected statistics
    }

    //info->CollectInfo();  // collected all the info and put into JSON mObject
    //delete info; info=NULL;
    util::PrintMsg("5");
    return retval;
}

} // namespace sssp
} // namespace app
} // namespace gunrock

/**
 * @brief Run test
 *
 * @tparam VertexId   Vertex identifier type
 * @tparam Value      Attribute type
 * @tparam SizeT      Graph size type
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 *
 * \return Elapsed run time in milliseconds
 */
/*template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool MARK_PREDECESSORS >
float runSSSP(GRGraph* output, SSSP_Parameter *parameter);
*/

/**
 * @brief Run test
 *
 * @tparam VertexId   Vertex identifier type
 * @tparam Value      Attribute type
 * @tparam SizeT      Graph size type
 *
 * @param[out] output    Pointer to output graph structure of the problem
 * @param[in]  parameter primitive-specific test parameters
 *
 * \return Elapsed run time in milliseconds
 */
/*template <
    typename    VertexId,
    typename    SizeT,
    typename    Value>
float markPredecessorsSSSP(GRGraph* output, SSSP_Parameter *parameter)
{
    if (parameter->mark_predecessors)
        return runSSSP<VertexId, SizeT, Value, true>(output, parameter);
    else
        return runSSSP<VertexId, SizeT, Value, false>(output, parameter);
}
*/

/**
 * @brief Dispatch function to handle configurations
 *
 * @param[out] grapho  Pointer to output graph structure of the problem
 * @param[in]  graphi  Pointer to input graph we need to process on
 * @param[in]  config  Primitive-specific configurations
 * @param[in]  data_t  Data type configurations
 * @param[in]  context ModernGPU context
 * @param[in]  streams CUDA stream
 *
 * \return Elapsed run time in milliseconds
 */
/*float dispatchSSSP(
    GRGraph*       grapho,
    const GRGraph* graphi,
    const GRSetup* config,
    const GRTypes  data_t,
    ContextPtr*    context,
    cudaStream_t*  streams)
{
    SSSP_Parameter *parameter = new SSSP_Parameter;
    parameter->iterations = config->num_iters;
    parameter->src = (long long*)malloc(sizeof(long long)*config->num_iters);
    parameter->context  = context;
    parameter->streams  = streams;
    parameter->g_quiet  = config -> quiet;
    parameter->num_gpus = config -> num_devices;
    parameter->gpu_idx  = config -> device_list;
    parameter->delta_factor = config -> delta_factor;
    parameter->traversal_mode = std::string(config -> traversal_mode);
    parameter->mark_predecessors  = config -> mark_predecessors;

    float elapsed_time;

    switch (data_t.VTXID_TYPE)
    {
    case VTXID_INT:
    {
        switch (data_t.SIZET_TYPE)
        {
        case SIZET_INT:
        {
            switch (data_t.VALUE_TYPE)
            {
            case VALUE_INT:    // template type = <int, int, int>
            {
                Csr<int, int, int> csr(false);
                csr.nodes = graphi->num_nodes;
                csr.edges = graphi->num_edges;
                csr.row_offsets    = (int*)graphi->row_offsets;
                csr.column_indices = (int*)graphi->col_indices;
                csr.edge_values    = (int*)graphi->edge_values;
                parameter->graph = &csr;

                // determine source vertex to start
                switch (config -> source_mode)
                {
                case randomize:
                {
                    parameter->src[0] = graphio::RandomNode(csr.nodes);
                    break;
                }
                case largest_degree:
                {
                    int max_deg = 0;
                    parameter->src[0] = csr.GetNodeWithHighestDegree(max_deg);
                    break;
                }
                case manually:
                {
                    parameter->src[0] = config -> source_vertex[0];
                    break;
                }
                default:
                {
                    parameter->src[0] = 0;
                    break;
                }
                }
                if (!parameter->g_quiet)
                {
                    printf(" source: %lld\n", (long long) parameter->src[0]);
                }

                elapsed_time = markPredecessorsSSSP<int, int, int>(grapho, parameter);

                // reset for free memory
                csr.row_offsets    = NULL;
                csr.column_indices = NULL;
                csr.edge_values    = NULL;
                break;
            }
            case VALUE_UINT:    // template type = <int, uint, int>
            {
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_FLOAT:
            {
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
    free(parameter->src);
    return elapsed_time;
}
*/

/*
 * @brief Entry of gunrock_sssp function
 *
 * @param[out] grapho Pointer to output graph structure of the problem
 * @param[in]  graphi Pointer to input graph we need to process on
 * @param[in]  config Gunrock primitive specific configurations
 * @param[in]  data_t Gunrock data type structure
 */
/*float gunrock_sssp(
    GRGraph*       grapho,
    const GRGraph* graphi,
    const GRSetup* config,
    const GRTypes  data_t)
{
    // GPU-related configurations
    int           num_gpus =    0;
    int           *gpu_idx = NULL;
    ContextPtr    *context = NULL;
    cudaStream_t  *streams = NULL;

    num_gpus = config -> num_devices;
    gpu_idx  = new int [num_gpus];
    for (int i = 0; i < num_gpus; ++i)
    {
        gpu_idx[i] = config -> device_list[i];
    }

    // Create streams and MordernGPU context for each GPU
    streams = new cudaStream_t[num_gpus * num_gpus * 2];
    context = new ContextPtr[num_gpus * num_gpus];
    if (!config -> quiet) { printf(" using %d GPUs:", num_gpus); }
    for (int gpu = 0; gpu < num_gpus; ++gpu)
    {
        if (!config -> quiet) { printf(" %d ", gpu_idx[gpu]); }
        util::SetDevice(gpu_idx[gpu]);
        for (int i = 0; i < num_gpus * 2; ++i)
        {
            int _i = gpu * num_gpus * 2 + i;
            util::GRError(cudaStreamCreate(&streams[_i]),
                          "cudaStreamCreate fialed.", __FILE__, __LINE__);
            if (i < num_gpus)
            {
                context[gpu * num_gpus + i] =
                    mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu],
                                                       streams[_i]);
            }
        }
    }
    if (!config -> quiet) { printf("\n"); }

    return dispatchSSSP(grapho, graphi, config, data_t, context, streams);
}*/

/*
 * @brief Simple interface take in CSR arrays as input
 *
 * @param[out] distances   Return shortest distance to source per nodes
 * @param[in]  num_nodes   Number of nodes of the input graph
 * @param[in]  num_edges   Number of edges of the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  source      Source to begin traverse
 */
/*template <
    typename VertexT,
    typename SizeT,
    typename GValueT,
    typename SSSPValueT>
float sssp(
          SSSPValueT *distances,
          VertexT    *preds,
    const SizeT       num_nodes,
    const SizeT       num_edges,
    const SizeT      *row_offsets,
    const VertexT    *col_indices,
    const GValueT    *edge_values,
    const int         num_runs,
          VertexT    *source,
    const bool        mark_preds)
{
    struct GRTypes data_t;          // primitive-specific data types
    data_t.VTXID_TYPE = VTXID_INT;  // integer vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;  // integer graph size type
    data_t.VALUE_TYPE = VALUE_INT;  // integer attributes type

    struct GRSetup *config = InitSetup(num_iters, source);  // primitive-specific configures
    config -> mark_predecessors = mark_preds;     // do not mark predecessors

    struct GRGraph *grapho = (struct GRGraph*)malloc(sizeof(struct GRGraph));
    struct GRGraph *graphi = (struct GRGraph*)malloc(sizeof(struct GRGraph));

    graphi->num_nodes   = num_nodes;  // setting graph nodes
    graphi->num_edges   = num_edges;  // setting graph edges
    graphi->row_offsets = (void*)&row_offsets[0];  // setting row_offsets
    graphi->col_indices = (void*)&col_indices[0];  // setting col_indices
    graphi->edge_values = (void*)&edge_values[0];  // setting edge_values

    float elapsed_time = gunrock_sssp(grapho, graphi, config, data_t);
    memcpy(distances, (int*)grapho->node_value1, num_nodes * sizeof(int));
    if (mark_preds)
        memcpy(preds, (int*)grapho->node_value2, num_nodes * sizeof(int));

    if (graphi) free(graphi);
    if (grapho) free(grapho);
    if (config) free(config);

    return elapsed_time;
}*/

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
