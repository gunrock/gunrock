// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_mf.cu
 *
 * @brief Simple test driver program for max-flow algorithm.
 */

#include <gunrock/app/mf/mf_app.cu>
#include <gunrock/app/test_base.cuh>

#define debug_aml(a...)
//#define debug_aml(a...) std::cerr << __FILE__ << ":" << __LINE__ << " " << \
    a << "\n";

using namespace gunrock;

/*****************************************************************************
* Main
*****************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct
{
    /**
     * @brief the actual main function, after type switching
     * @tparam VertexT	  Type of vertex identifier
     * @tparam SizeT      Type of graph size, i.e. type of edge identifier
     * @tparam ValueT     Type of edge values
     * @param  parameters Command line parameters
     * @param  v,s,val    Place holders for type deduction
     * \return cudaError_t error message(s), if any
     */
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(util::Parameters &parameters, VertexT v, SizeT s, 
	    ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT, 
	  graph::HAS_EDGE_VALUES | graph::HAS_CSR> GraphT;
	typedef typename GraphT::CsrT CsrT;
        cudaError_t retval = cudaSuccess;
	bool quick = parameters.Get<bool>("quick");
        bool quiet = parameters.Get<bool>("quiet");
	
	//
	// Load Graph
	//
	bool undirected = parameters.Get<int>("undirected");
        util::CpuTimer cpu_timer; cpu_timer.Start();
	debug_aml("Start Load Graph");
        
	GraphT d_graph;
	parameters.Set<int>("undirected", 0);
        GUARD_CU(graphio::LoadGraph(parameters, d_graph));
	
	GraphT u_graph;
	parameters.Set<int>("undirected", 1);
	GUARD_CU(graphio::LoadGraph(parameters, u_graph));

	cpu_timer.Stop();

	parameters.Set("load-time", cpu_timer.ElapsedMillis());
	debug_aml("load-time is " << cpu_timer.ElapsedMillis());


	if (parameters.Get<VertexT>("source") == 
		util::PreDefinedValues<VertexT>::InvalidValue){
	    parameters.Set("source", 0);
	}
	if (parameters.Get<VertexT>("sink") == 
		util::PreDefinedValues<VertexT>::InvalidValue){
	    parameters.Set("sink", d_graph.nodes-1);
	}

	VertexT source = parameters.Get<VertexT>("source");
	VertexT sink = parameters.Get<VertexT>("sink");

	debug_aml("Directed graph:");
	debug_aml("number of edges " << d_graph.edges);
	debug_aml("number of nodes " << d_graph.nodes);
	debug_aml("source " << source);
	debug_aml("sink " << sink);

	for (auto u = 0; u < d_graph.nodes; ++u){
	    debug_aml("vertex " << u);
	    auto e_start = d_graph.CsrT::GetNeighborListOffset(u);
	    auto num_neighbors = d_graph.CsrT::GetNeighborListLength(u);
	    auto e_end = e_start + num_neighbors;
	    for (auto e = e_start; e < e_end; ++e)
	    {
		auto v = d_graph.CsrT::GetEdgeDest(e);
		auto cap = d_graph.edge_values[e];
		debug_aml("edge (" << u << ", " << v << ") cap = " << cap);
	    }
	}
	ValueT* flow_edge = (ValueT*)malloc(sizeof(ValueT)*u_graph.edges);
	SizeT* reverse = (SizeT*)malloc(sizeof(SizeT)*u_graph.edges);

	// Initialize reverse array.
	for (auto u = 0; u < u_graph.nodes; ++u)
	{
	    auto e_start = u_graph.CsrT::GetNeighborListOffset(u);
	    auto num_neighbors = u_graph.CsrT::GetNeighborListLength(u);
	    auto e_end = e_start + num_neighbors;
	    for (auto e = e_start; e < e_end; ++e)
	    {
		auto v = u_graph.CsrT::GetEdgeDest(e);
		auto f_start = u_graph.CsrT::GetNeighborListOffset(v);
		auto num_neighbors2 = u_graph.CsrT::GetNeighborListLength(v);
		auto f_end = f_start + num_neighbors2;
		for (auto f = f_start; f < f_end; ++f)
		{
		    auto z = u_graph.CsrT::GetEdgeDest(f);
		    if (z == u)
		    {
			reverse[e] = f;
			reverse[f] = e;
		    }
		}
	    }
	}

	// Correct capacity values on reverse edges
	for (auto u = 0; u < d_graph.nodes; ++u)
	{
	    auto e_start = d_graph.CsrT::GetNeighborListOffset(u);
	    auto num_neighbors = d_graph.CsrT::GetNeighborListLength(u);
	    auto e_end = e_start + num_neighbors;
	    for (auto e = e_start; e < e_end; ++e)
	    {
		auto v = d_graph.CsrT::GetEdgeDest(e);
		// Looking for the reverse edge in undirected graph
		auto f_start = u_graph.CsrT::GetNeighborListOffset(v);
		auto num_neighbors2 = u_graph.CsrT::GetNeighborListLength(v);
		auto f_end = f_start + num_neighbors2;
		for (auto f = f_start; f < f_end; ++f)
		{
		    auto z = u_graph.CsrT::GetEdgeDest(f);
		    if (z == u)
		    {
			u_graph.CsrT::edge_values[f] = 0;
		    }
		}
	    }
	}

      	debug_aml("Undirected graph:");
	debug_aml("number of edges " << u_graph.edges);
	debug_aml("number of nodes " << u_graph.nodes);
	debug_aml("source " << source);
	debug_aml("sink " << sink);

	for (auto u = 0; u < u_graph.nodes; ++u){
	    debug_aml("vertex " << u);
	    auto e_start = u_graph.CsrT::GetNeighborListOffset(u);
	    auto num_neighbors = u_graph.CsrT::GetNeighborListLength(u);
	    auto e_end = e_start + num_neighbors;
	    for (auto e = e_start; e < e_end; ++e)
	    {
		auto v = u_graph.CsrT::GetEdgeDest(e);
		auto cap = u_graph.edge_values[e];
		debug_aml("edge (" << u << ", " << v << ") cap = " << cap);
	    }
	}

	//
        // Compute reference CPU max flow algorithm.
	//
	GraphT graph = u_graph;
        ValueT max_flow;
	
	util::PrintMsg("______CPU reference algorithm______", true);
	double elapsed = app::mf::CPU_Reference
	    (parameters, graph, source, sink, max_flow, reverse, flow_edge);
        util::PrintMsg("------------------------------------\n\elapsed: " + 
		std::to_string(elapsed) + " ms, max flow = " +
		std::to_string(max_flow), true);
	
        std::vector<std::string> switches{"advance-mode"};
	GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
	[flow_edge, reverse](util::Parameters &parameters, GraphT &graph)
	{
	debug_aml("go to RunTests");
	return app::mf::RunTests(parameters, graph, reverse, flow_edge);
	}));

	// Clean up
	free(flow_edge);
	free(reverse);
	
        return retval;
    }
};

int main(int argc, char** argv)
{
    debug_aml("Main: start");
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test mf");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::mf::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());
    debug_aml("Main: parameters checked - ok");

    return app::Switch_Types<
        app::VERTEXT_U32B | 
        app::SIZET_U32B | 
        app::VALUET_F64B | 
	app::DIRECTED >
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

