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
//#define debug_aml(a...) {printf(a); printf("\n");}

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
        util::CpuTimer cpu_timer; cpu_timer.Start();
	debug_aml("Start Load Graph");
      
	bool undirected;
	parameters.Get("undirected", undirected);
	if (undirected){
	    debug_aml("graph is undirected");
	}else{
	    debug_aml("graph is directed");
	}

	GraphT d_graph;
	if (not undirected){
	    debug_aml("Load directed graph");
	    parameters.Set<int>("remove-duplicate-edges", false);
	    GUARD_CU(graphio::LoadGraph(parameters, d_graph));
	}

	debug_aml("Load undirected graph");
	GraphT u_graph;
	parameters.Set<int>("undirected", 1);
	parameters.Set<int>("remove-duplicate-edges", true);
        GUARD_CU(graphio::LoadGraph(parameters, u_graph));
	
	cpu_timer.Stop();

	parameters.Set("load-time", cpu_timer.ElapsedMillis());
	debug_aml("load-time is %lf",cpu_timer.ElapsedMillis());

	if (parameters.Get<VertexT>("source") == 
		util::PreDefinedValues<VertexT>::InvalidValue){
	    parameters.Set("source", 0);
	}
	if (parameters.Get<VertexT>("sink") == 
		util::PreDefinedValues<VertexT>::InvalidValue){
	    parameters.Set("sink", u_graph.nodes-1);
	}

	VertexT source = parameters.Get<VertexT>("source");
	VertexT sink = parameters.Get<VertexT>("sink");

	if (not undirected){
	    debug_aml("Directed graph:");
	    debug_aml("number of edges %d", d_graph.edges);
	    debug_aml("number of nodes %d", d_graph.nodes);
	}

	debug_aml("Undirected graph:");
	debug_aml("number of edges %d", u_graph.edges);
	debug_aml("number of nodes %d", u_graph.nodes);

	ValueT* flow_edge = (ValueT*)malloc(sizeof(ValueT)*u_graph.edges);
	SizeT* reverse	  = (SizeT*)malloc(sizeof(SizeT)*u_graph.edges);

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
			break;
		    }
		}
	    }
	}

	if (not undirected){
	    // Correct capacity values on reverse edges
	    for (auto u = 0; u < u_graph.nodes; ++u)
	    {
		auto e_start = u_graph.CsrT::GetNeighborListOffset(u);
		auto num_neighbors = u_graph.CsrT::GetNeighborListLength(u);
		auto e_end = e_start + num_neighbors;
		debug_aml("vertex %d\nnumber of neighbors %d", u, 
			num_neighbors);
		for (auto e = e_start; e < e_end; ++e)
		{
		    u_graph.CsrT::edge_values[e] = (ValueT)0;
		    auto v = u_graph.CsrT::GetEdgeDest(e);
		    // Looking for edge u->v in directed graph
		    auto f_start = d_graph.CsrT::GetNeighborListOffset(u);
		    auto num_neighbors2 = 
			d_graph.CsrT::GetNeighborListLength(u);
		    auto f_end = f_start + num_neighbors2;
		    for (auto f = f_start; f < f_end; ++f)
		    {
			auto z = d_graph.CsrT::GetEdgeDest(f);
			if (z == v and d_graph.CsrT::edge_values[f] > 0)
			{
			    u_graph.CsrT::edge_values[e]  = 
				d_graph.CsrT::edge_values[f];
			    debug_aml("edge (%d, %d) cap = %lf\n", u, v, \
				    u_graph.CsrT::edge_values[e]);
			    break;
			}
		    }
		}
	    }
	}
/*
	ValueT** rGraph = (ValueT**)malloc(sizeof(ValueT*)*u_graph.nodes);
	for (auto x = 0; x < u_graph.nodes; ++x){
	    rGraph[x] = (ValueT*)malloc(sizeof(ValueT)*u_graph.nodes);
	    for (auto y = 0; y < u_graph.nodes; ++y){
		rGraph[x][y] = (ValueT)0;
	    }
	}
	for (auto x = 0; x < u_graph.nodes; ++x)
	{
	    auto e_start = u_graph.CsrT::GetNeighborListOffset(x);
	    auto num_neighbors = u_graph.CsrT::GetNeighborListLength(x);
	    auto e_end = e_start + num_neighbors;
	    for (auto e = e_start; e < e_end; ++e){
		auto y = u_graph.CsrT::GetEdgeDest(e);
		auto f = u_graph.CsrT::edge_values[e];
		rGraph[x][y] = f;
	    }
	}

	FILE * rgraph_file = fopen("rgraph_output", "w");
	printf("number of nodes %d\n", u_graph.nodes);
	fprintf(rgraph_file, "{");
	for (auto n = 0; n < u_graph.nodes; ++n){
	    fprintf(rgraph_file, "{");
	    for (auto m = 0; m < u_graph.nodes; ++m){
		fprintf(rgraph_file, "%.0lf", rGraph[n][m]);
		if (m == u_graph.nodes-1){
		    fprintf(rgraph_file, " ");
		}else{
		    fprintf(rgraph_file, ", ");
		}
	    }
	    fprintf(rgraph_file, "}\n");
	}
	fprintf(rgraph_file, "}");
	fclose(rgraph_file);
*/

	//
        // Compute reference CPU max flow algorithm.
	//
        ValueT max_flow;
	
	util::PrintMsg("______CPU reference algorithm______", true);
	double elapsed = app::mf::CPU_Reference
	    (parameters, u_graph, source, sink, max_flow, reverse, flow_edge);
        util::PrintMsg("-----------------------------------\nElapsed: " + 
		std::to_string(elapsed) + " ms\nMax flow CPU = " +
		std::to_string(max_flow), true);
	
        std::vector<std::string> switches{"advance-mode"};
	GUARD_CU(app::Switch_Parameters(parameters, u_graph, switches,
	[flow_edge, reverse](util::Parameters &parameters, GraphT &u_graph)
	{
	  debug_aml("go to RunTests");
	  if (flow_edge)
	  {
	    for (int i=0; i<u_graph.edges; ++i)
	    {
	      debug_aml("%s:%d flow_edge[%d] = %lf", 
		__FILE__, __LINE__, i, flow_edge[i]);
	    }
	  }
	  return app::mf::RunTests(parameters, u_graph, reverse, flow_edge);
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
	app::DIRECTED | app::UNDIRECTED >
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

