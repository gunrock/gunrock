// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_gtf.cu
 *
 * @brief Simple test driver program for max-flow algorithm.
 */

#include <gunrock/app/gtf/gtf_app.cu>
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
        GraphT u_graph;
    	bool undirected;
    	parameters.Get("undirected", undirected);

    	if (undirected) {
    	    debug_aml("graph is undirected");
            debug_aml("Load undirected graph");
        	//parameters.Set<int>("undirected", 1);
        	parameters.Set<bool>("remove-duplicate-edges", true);
            GUARD_CU(graphio::LoadGraph(parameters, u_graph));

    	} else {
    	    debug_aml("graph is directed");
            GraphT d_graph;
            debug_aml("Load directed graph");
            //parameters.Set<int>("undirected", 0);
    	    //parameters.Set<bool>("remove-duplicate-edges", false);
    	    GUARD_CU(graphio::LoadGraph(parameters, d_graph));

            debug_aml("Directed graph:");
    	    debug_aml("number of edges %d", d_graph.edges);
    	    debug_aml("number of nodes %d", d_graph.nodes);

            GUARD_CU(graphio::MakeUndirected(d_graph, u_graph, false));
            GUARD_CU(mf::CorrectReverseCapacities(
                d_graph.csr(), u_graph.csr()));

            GUARD_CU(d_graph.Release());
        }

        util::Array1D<SizeT, ValueT> weights;
        std::string weights_filename = parameters.Get<std::string>("weights");
        GUARD_CU(weights.Read(weights_filename));

        GraphT graph;
        GUARD_CU(gtf::AddSourceSink(u_graph.csr(), weights, graph.csr()));
        GUARD_CU(u_graph.Release());

    	cpu_timer.Stop();
    	parameters.Set("load-time", cpu_timer.ElapsedMillis());
    	debug_aml("load-time is %lf",cpu_timer.ElapsedMillis());

        GUARD_CU(parameters.Set("source", graph.nodes - 2));
        GUARD_CU(parameters.Set("sink"  , graph.nodes - 1));

    	debug_aml("Undirected graph:");
    	debug_aml("number of edges %d", graph.edges);
    	debug_aml("number of nodes %d", graph.nodes);

        util::Array1D<SizeT, SizeT> reverse_edges;
        reverse_edges.SetName("reverse_edges");
        GUARD_CU(reverse_edges.Allocate(graph.edges, util::HOST));

    	GUARD_CU(mf::InitReverse(graph, reverse_edges));

	    //
        // Compute reference CPU GTF algorithm.
	    //
    	util::PrintMsg("______CPU reference algorithm______", true);
    	double elapsed = app::gtf::CPU_Reference
    	    (parameters, graph, reverse_edges);
        util::PrintMsg("-----------------------------------\n"
            "Elapsed: " + std::to_string(elapsed) + " ms", true);

        std::vector<std::string> switches{"advance-mode"};
    	GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
    	[reverse_edges](util::Parameters &parameters, GraphT &graph)
    	{
    	    //return app::gtf::RunTests(parameters, graph, reverse_edges);
    	}));

    	// Clean up
    	GUARD_CU(reverse_edges.Release());
        GUARD_CU(graph.Release());
        
        return retval;
    }
};

int main(int argc, char** argv)
{
    debug_aml("Main: start");
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test gtf");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::gtf::UseParameters(parameters));
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
