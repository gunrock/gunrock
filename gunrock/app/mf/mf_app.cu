// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file mf_app.cu
 *
 * @brief maxflow (mf) application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/type_limits.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// MF includes
#include <gunrock/app/mf/mf_enactor.cuh>
#include <gunrock/app/mf/mf_test.cuh>

//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a);\
    printf("\n");}
#define debug_aml(a...)

namespace gunrock {
namespace app {
namespace mf {

cudaError_t UseParameters(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(UseParameters_app    (parameters));
    GUARD_CU(UseParameters_problem(parameters));
    GUARD_CU(UseParameters_enactor(parameters));

    GUARD_CU(parameters.Use<uint32_t>(
	"source",
	util::REQUIRED_ARGUMENT,
	util::PreDefinedValues<uint32_t>::InvalidValue,
	"<Vertex-ID|random|largestdegree> The source vertex\n"
	"\tIf random, randomly select non-zero degree vertex;\n"
	"\tIf largestdegree, select vertex with largest degree",
	__FILE__, __LINE__));

    GUARD_CU(parameters.Use<uint32_t>(
	"sink",
	util::REQUIRED_ARGUMENT,
	util::PreDefinedValues<uint32_t>::InvalidValue,
	"<Vertex-ID|random|largestdegree> The source vertex\n"
	"\tIf random, randomly select non-zero degree vertex;\n"
	"\tIf largestdegree, select vertex with largest degree",
	__FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
	"seed",
	util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
	util::PreDefinedValues<int>::InvalidValue,
	"seed to generate random sources or sink",
	__FILE__, __LINE__)); 
    return retval;
}

/**
 * @brief Run mf tests
 * @tparam     GraphT	  Type of the graph
 * @tparam     ValueT	  Type of the capacity on edges
 * @tparam     VertexT	  Type of vertex
 * @param[in]  parameters Excution parameters
 * @param[in]  graph	  Input graph
 * @param[in]  ref_flow	  Reference flow on edges
 * @param[in]  target	  Whether to perform the mf
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename ValueT, typename VertexT>
cudaError_t RunTests(
    util::Parameters  &parameters,
    GraphT	      &graph,
    VertexT	      *h_reverse,
    ValueT	      *ref_flow,
    util::Location    target = util::DEVICE)
{
    debug_aml("RunTests starts");
    cudaError_t retval = cudaSuccess;
    typedef Problem<GraphT>	      ProblemT;
    typedef Enactor<ProblemT>	      EnactorT;

    util::CpuTimer total_timer;	total_timer.Start();
    util::CpuTimer cpu_timer;	cpu_timer.Start(); 

    // parse configurations from parameters
    bool quiet_mode	    = parameters.Get<bool>("quiet");
    int  num_runs	    = parameters.Get<int >("num-runs");
    std::string validation  = parameters.Get<std::string>("validation");
    VertexT source	    = parameters.Get<VertexT>("source");
    VertexT sink	    = parameters.Get<VertexT>("sink");
    debug_aml("source %d, sink %d, quite_mode %d, num-runs %d", source, sink,
	    quiet_mode, num_runs);

    util::Info info("MF", parameters, graph); // initialize Info structure

    // Allocate host-side array (for both reference and GPU-computed results)
    // ... for function Extract
    ValueT *h_flow   = (ValueT*)malloc(sizeof(ValueT)*graph.edges);
    
    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem(parameters);
    EnactorT enactor;
    GUARD_CU(problem.Init(graph, target));
    GUARD_CU(enactor.Init(problem, target));

    cpu_timer.Stop();
    parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

    // perform the MF algorithm
    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        GUARD_CU(problem.Reset(graph, h_reverse, target));
        GUARD_CU(enactor.Reset(source, target));

        util::PrintMsg("______GPU PushRelabel algorithm____", !quiet_mode);

        cpu_timer.Start();
        GUARD_CU(enactor.Enact());
        cpu_timer.Stop();
        info.CollectSingleRun(cpu_timer.ElapsedMillis());

        util::PrintMsg("-----------------------------------\nRun "
            + std::to_string(run_num) + ", elapsed: "
            + std::to_string(cpu_timer.ElapsedMillis()) +
            " ms, #iterations = "
            + std::to_string(enactor.enactor_slices[0]
                .enactor_stats.iteration), !quiet_mode);
        if (validation == "each")
        {
            GUARD_CU(problem.Extract(h_flow));
            int num_errors = app::mf::Validate_Results(parameters, graph, 
		    source, sink, h_flow, h_reverse, ref_flow, quiet_mode);
        }
    }
    
    // Copy out results
    cpu_timer.Start();
    if (validation == "last")
    {
	GUARD_CU(problem.Extract(h_flow));
	/*for (int i=0; i<graph.edges; ++i){
	    if (ref_flow){
		debug_aml("h_flow[%d]=%lf, ref_flow[%d] = %lf", 
			  i, h_flow[i], i, ref_flow[i]);
	    }
	}*/
        int num_errors = app::mf::Validate_Results(parameters, graph, 
		source, sink, h_flow, h_reverse, ref_flow, quiet_mode);
    }

    // Compute running statistics
    //info.ComputeTraversalStats(enactor, h_flow);
    // Display_Memory_Usage(problem);
    #ifdef ENABLE_PERFORMANCE_PROFILING
        //Display_Performance_Profiling(enactor);
    #endif

    // Clean up
    GUARD_CU(enactor.Release(target));
    GUARD_CU(problem.Release(target));
    delete[] h_flow; 
    h_flow = NULL;

    cpu_timer.Stop(); 
    total_timer.Stop();

    info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
    
    return retval;
}

} // namespace mf
} // namespace app
} // namespace gunrock

/*
 * @brief Entry of gunrock_maxflow function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the capacity/flow/excess
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] flow	  Return 
 * @param[out] maxflow	  Return 
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_mf(
    gunrock::util::Parameters &parameters,
    GraphT &graph,
    ValueT *flow,
    ValueT &maxflow
    )
{
    typedef typename GraphT::VertexT		VertexT;
    typedef gunrock::app::mf::Problem<GraphT>	ProblemT;
    typedef gunrock::app::mf::Enactor<ProblemT> EnactorT;
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
    int source = parameters.Get<VertexT>("source");
    int sink = parameters.Get<VertexT>("sink");

    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        problem.Reset(target);
        enactor.Reset(target);

        cpu_timer.Start();
        enactor.Enact(source, sink);
        cpu_timer.Stop();

        total_time += cpu_timer.ElapsedMillis();
        problem.Extract(flow);
    }

    enactor.Release(target);
    problem.Release(target);
    return total_time;
}

/*
 * @brief Simple interface  take in graph as CSR format
 * @param[in]  num_nodes    Number of veritces in the input graph
 * @param[in]  num_edges    Number of edges in the input graph
 * @param[in]  row_offsets  CSR-formatted graph input row offsets
 * @param[in]  col_indices  CSR-formatted graph input column indices
 * @param[in]  capacity	    CSR-formatted graph input edge weights
 * @param[in]  num_runs     Number of runs to perform mf
 * @param[in]  source	    Source to push flow towards the sink
 * @param[out] flow	    Return flow calculated on edges
 * @param[out] maxflow	    Return maxflow value
 * \return     double       Return accumulated elapsed times for all runs
 */
template <
    typename VertexT  = uint32_t,
    typename SizeT    = uint32_t,
    typename ValueT   = double>
float mf(
	const SizeT   num_nodes,
	const SizeT   num_edges,
	const SizeT   *row_offsets,
	const VertexT *col_indices,
	const ValueT  capacity,
	const int     num_runs,
	VertexT	      source,
	VertexT	      sink,
	ValueT	      *flow,
	ValueT	      &maxflow
	)
{
    // TODO: change to other graph representation, if not using CSR
    typedef typename gunrock::app::TestGraph<VertexT, SizeT, ValueT,
        gunrock::graph::HAS_EDGE_VALUES | gunrock::graph::HAS_COO>  GraphT;
    typedef typename GraphT::CooT				    CooT;
    typedef typename GraphT::CsrT				    CsrT;

    // Setup parameters
    gunrock::util::Parameters parameters("mf");
    gunrock::graphio::UseParameters(parameters);
    gunrock::app::mf::UseParameters(parameters);
    gunrock::app::UseParameters_test(parameters);
    parameters.Parse_CommandLine(0, NULL);
    parameters.Set("graph-type", "by-pass");
    parameters.Set("num-runs", num_runs);
    parameters.Set("source", source);
    parameters.Set("sink", sink);

    bool quiet = parameters.Get<bool>("quiet");
    CsrT csr;
    // Assign pointers into gunrock graph format
    csr.Allocate(num_nodes, num_edges, gunrock::util::HOST);
    csr.row_offsets   .SetPointer(row_offsets,gunrock::util::HOST);
    csr.column_indices.SetPointer(col_indices,gunrock::util::HOST);
    csr.capacity      .SetPointer(capacity,   gunrock::util::HOST);

    gunrock::util::Location target = gunrock::util::HOST;
    CooT graph;
    graph.FromCsr(csr, target, 0, quiet, true);
    csr.Release();
    gunrock::graphio::LoadGraph(parameters, graph);

    // Run the MF
    double elapsed_time = gunrock_mf(parameters, graph, flow, maxflow);

    // Cleanup
    graph.Release();

    return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

