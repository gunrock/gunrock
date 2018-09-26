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
						      graph::HAS_COO> GraphT;
	typedef typename GraphT::CooT CooT;
        cudaError_t retval = cudaSuccess;
	bool quick = parameters.Get<bool>("quick");
        bool quiet = parameters.Get<bool>("quiet");


	//
	// Load Graph
	//
        GraphT graph;
        util::CpuTimer cpu_timer; cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        
	//FOR DEBUG: force edge values to be 1
        for (SizeT e=0; e < graph.edges; e++){
	    graph.CooT::edge_values[e] = 1;
	}
        
	cpu_timer.Stop();
        
	parameters.Set("load-time", cpu_timer.ElapsedMillis());

	VertexT source = parameters.Get<VertexT>("source");
	VertexT sink = parameters.Get<VertexT>("sink");

	//
        // Compute reference CPU max flow algorithm.
	//
        ValueT max_flow;
	ValueT** flow_edges = (ValueT**)malloc(sizeof(ValueT*)*graph.nodes);
	for (VertexT i=0; i<graph.nodes; ++i){
	    flow_edges[i] = (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
	    for (VertexT j=0; j<graph.nodes; ++j){
		flow_edges[i][j] = 0;
	    }
	}
        util::PrintMsg("______CPU reference algorithm______", true);
	double elapsed = app::mf::CPU_Reference
	    (parameters, graph, source, sink, max_flow, flow_edges);
        util::PrintMsg("------------------------------------\n\
		elapsed: " + std::to_string(elapsed) + 
		" ms, max flow = " + std::to_string(max_flow), true);

	
        std::vector<std::string> switches{"advance-mode"};
	GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
	[flow_edges](util::Parameters &parameters, GraphT &graph)
	{
	return app::mf::RunTests(parameters, graph, flow_edges);
	}));

	// Clean up
	for (VertexT i=0; i<graph.nodes; ++i)
	    free(flow_edges[i]);
	free(flow_edges);
	
        return retval;
    }
};

int main(int argc, char** argv)
{
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

    return app::Switch_Types<
        app::VERTEXT_U32B | 
        app::SIZET_U32B | 
        app::VALUET_U32B | 
	app::DIRECTED | app::UNDIRECTED >
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

