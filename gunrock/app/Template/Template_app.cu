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
#include <gunrock/app/test_base.cuh>

// single-source shortest path includes
#include <gunrock/app/Template/Template_enactor.cuh>
#include <gunrock/app/Template/Template_test.cuh>

namespace gunrock {
namespace app {
// TODO: change the space name
namespace Template {

cudaError_t UseParameters(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(UseParameters_app    (parameters));
    GUARD_CU(UseParameters_problem(parameters));
    GUARD_CU(UseParameters_enactor(parameters));

    // TODO: add app specific parameters, e.g.:
    // GUARD_CU(parameters.Use<std::string>(
    //    "src",
    //    util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    "0",
    //    "<Vertex-ID|random|largestdegree> The source vertices\n"
    //    "\tIf random, randomly select non-zero degree vertices;\n"
    //    "\tIf largestdegree, select vertices with largest degrees",
    //    __FILE__, __LINE__));
    //
    // GUARD_CU(parameters.Use<int>(
    //    "src-seed",
    //    util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
    //    util::PreDefinedValues<int>::InvalidValue,
    //    "seed to generate random sources",
    //    __FILE__, __LINE__));

    return retval;
}

/**
 * @brief Run SSSP tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  ref_distances Reference distances
 * @param[in]  target        Whether to perform the SSSP
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
cudaError_t RunTests(
    util::Parameters &parameters,
    GraphT           &graph,
    // TODO: add problem specific reference results, e.g.:
    // ValueT **ref_distances = NULL,
    util::Location target = util::DEVICE)
{
    cudaError_t retval = cudaSuccess;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef Problem<GraphT  > ProblemT;
    typedef Enactor<ProblemT> EnactorT;
    util::CpuTimer    cpu_timer, total_timer;
    cpu_timer.Start(); total_timer.Start();

    // parse configurations from parameters
    bool quiet_mode = parameters.Get<bool>("quiet");
    int  num_runs   = parameters.Get<int >("num-runs");
    std::string validation = parameters.Get<std::string>("validation");
    util::Info info("Template", parameters, graph); // initialize Info structure

    // TODO: get problem specific inputs, e.g.:
    // std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
    // int  num_srcs   = srcs   .size();

    // Allocate host-side array (for both reference and GPU-computed results)
    // TODO: allocate problem specific host data, e.g.:
    // ValueT  *h_distances = new ValueT[graph.nodes];

    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem(parameters);
    EnactorT enactor;
    GUARD_CU(problem.Init(graph  , target));
    GUARD_CU(enactor.Init(problem, target));
    cpu_timer.Stop();
    parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

    // perform the algorithm
    // TODO: Declear problem specific variables, e.g.:
    // VertexT src;
    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        // TODO: assign problem specific variables, e.g.:
        // src = srcs[run_num % num_srcs];
        GUARD_CU(problem.Reset(/*src,*/ target));
        GUARD_CU(enactor.Reset(/*src,*/ target));
        util::PrintMsg("__________________________", !quiet_mode);

        cpu_timer.Start();
        GUARD_CU(enactor.Enact(/*src*/));
        cpu_timer.Stop();
        info.CollectSingleRun(cpu_timer.ElapsedMillis());

        util::PrintMsg("--------------------------\nRun "
            + std::to_string(run_num) + " elapsed: "
            + std::to_string(cpu_timer.ElapsedMillis()) +
            //" ms, src = " + std::to_string(src) +
            ", #iterations = "
            + std::to_string(enactor.enactor_slices[0]
                .enactor_stats.iteration), !quiet_mode);
        if (validation == "each")
        {
            // TODO: fill in problem specific data, e.g.:
            GUARD_CU(problem.Extract(/*h_distances*/));
            SizeT num_errors = app::Template::Validate_Results(
                parameters, graph,
                // TODO: add problem specific data for validation, e.g.:
                // src, h_distances,
                // ref_distances == NULL ? NULL : ref_distances[run_num % num_srcs],
                // NULL,
                false);
        }
    }

    cpu_timer.Start();
    // Copy out results
    // TODO: fill in problem specific data, e.g.:
    GUARD_CU(problem.Extract(/*h_distances*/));
    if (validation == "last")
    {
        SizeT num_errors = app::Template::Validate_Results(
            parameters, graph,
            // TODO: place problem specific data and result for validation, e.g.:
            // src, h_distances,
            // ref_distances == NULL ? NULL : ref_distances[(num_runs -1) % num_srcs],
            // NULL,
            true);
    }

    // compute running statistics
    // TODO: change NULL to problem specific per-vertex visited marker, e.g. h_distances
    info.ComputeTraversalStats(enactor, (VertexT*)NULL);
    //Display_Memory_Usage(problem);
    #ifdef ENABLE_PERFORMANCE_PROFILING
        //Display_Performance_Profiling(enactor);
    #endif

    // Clean up
    GUARD_CU(enactor.Release(target));
    GUARD_CU(problem.Release(target));
    // TODO: Release problem specific data, e.g.:
    // delete[] h_distances  ; h_distances   = NULL;
    cpu_timer.Stop(); total_timer.Stop();

    info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
    return retval;
}

} // namespace Template
} // namespace app
} // namespace gunrock

/*
 * @brief Entry of gunrock_template function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the distances
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] distances  Return shortest distance to source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_Template(
    gunrock::util::Parameters &parameters,
    GraphT &graph
    // TODO: add problem specific outputs, e.g.:
    //ValueT **distances
    )
{
    typedef typename GraphT::VertexT VertexT;
    typedef gunrock::app::Template::Problem<GraphT  > ProblemT;
    typedef gunrock::app::Template::Enactor<ProblemT> EnactorT;
    gunrock::util::CpuTimer cpu_timer;
    gunrock::util::Location target = gunrock::util::DEVICE;
    double total_time = 0;
    if (parameters.UseDefault("quiet"))
        parameters.Set("quiet", true);

    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem(parameters);
    EnactorT enactor;
    problem.Init(graph  , target);
    enactor.Init(problem, target);

    int num_runs = parameters.Get<int>("num-runs");
    // TODO: get problem specific inputs, e.g.:
    // std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("srcs");
    // int num_srcs = srcs.size();
    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        // TODO: problem specific inputs, e.g.:
        // int src_num = run_num % num_srcs;
        // VertexT src = srcs[src_num];
        problem.Reset(/*src,*/ target);
        enactor.Reset(/*src,*/ target);

        cpu_timer.Start();
        enactor.Enact(/*src*/);
        cpu_timer.Stop();

        total_time += cpu_timer.ElapsedMillis();
        // TODO: extract problem specific data, e.g.:
        problem.Extract(/*distances[src_num]*/);
    }

    enactor.Release(target);
    problem.Release(target);
    // TODO: problem specific clean ups, e.g.:
    // srcs.clear();
    return total_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform SSSP
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[out] distances   Return shortest distance to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
template <
    typename VertexT = int,
    typename SizeT   = int,
    typename GValueT = unsigned int,
    typename TValueT = GValueT>
float Template(
    const SizeT        num_nodes,
    const SizeT        num_edges,
    const SizeT       *row_offsets,
    const VertexT     *col_indices,
    const GValueT     *edge_values,
    const int          num_runs
    // TODO: add problem specific inputs and outputs, e.g.:
    //      VertexT     *sources,
    //      SSSPValueT **distances
    )
{
    // TODO: change to other graph representation, if not using CSR
    typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
        gunrock::graph::HAS_EDGE_VALUES | gunrock::graph::HAS_CSR>
        GraphT;
    typedef typename GraphT::CsrT CsrT;

    // Setup parameters
    gunrock::util::Parameters parameters("Template");
    gunrock::graphio::UseParameters(parameters);
    gunrock::app::Template::UseParameters(parameters);
    gunrock::app::UseParameters_test(parameters);
    parameters.Parse_CommandLine(0, NULL);
    parameters.Set("graph-type", "by-pass");
    parameters.Set("num-runs", num_runs);
    // TODO: problem specific inputs, e.g.:
    // std::vector<VertexT> srcs;
    // for (int i = 0; i < num_runs; i ++)
    //     srcs.push_back(sources[i]);
    // parameters.Set("srcs", srcs);

    bool quiet = parameters.Get<bool>("quiet");
    GraphT graph;
    // Assign pointers into gunrock graph format
    // TODO: change to other graph representation, if not using CSR
    graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
    graph.CsrT::row_offsets   .SetPointer(row_offsets, gunrock::util::HOST);
    graph.CsrT::column_indices.SetPointer(col_indices, gunrock::util::HOST);
    graph.CsrT::edge_values   .SetPointer(edge_values, gunrock::util::HOST);
    graph.FromCsr(graph.csr(), true, quiet);
    gunrock::graphio::LoadGraph(parameters, graph);

    // Run the Template
    // TODO: add problem specific outputs, e.g.
    double elapsed_time = gunrock_Template(parameters, graph /*, distances*/);

    // Cleanup
    graph.Release();
    // TODO: problem specific cleanup
    // srcs.clear();

    return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
