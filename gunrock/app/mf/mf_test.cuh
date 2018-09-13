// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @fille
 * mf_test.cu
 *
 * @brief Test related functions for Max Flow algorithm.
 */

#pragma once

#ifdef BOOST_FOUND
    // Boost includes for CPU Push Relabel Max Flow reference algorithms
    #include <boost/config.hpp>
    #include <string>
    #include <boost/graph/push_relabel_max_flow.hpp>
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/read_dimacs.hpp>
    #include <boost/property_map/property_map.hpp>
#endif

namespace gunrock {
namespace app {
namespace mf {

/*****************************************************************************
 * Housekeeping Routines
 ****************************************************************************/

/**
 * @brief Displays the MF result
 *
 * @tparam ValueT     Type of capacity/flow/excess
 * @tparam VertxeT    Type of vertex 
 * 
 * @param[in] h_flow  Flow calculated on edges 
 * @param[in] source  Index of source vertex
 * @param[in] nodes   Number of nodes
 */
template<typename ValueT, typename VertexT>
void DisplaySolution(ValueT** h_flow, VertexT source, VertexT nodes)
{
    ValueT flow_outcoming_source = 0;
    for (VertexT v = 0; v < nodes; ++v)
    {
	ValueT flow_source_v = h_flow[source][v];
	if (util::isValid(flow_source_v))
	    flow_outcoming_source += flow_source_v;
    }

    util::PrintMsg("The maximum amount of flow that is feasible to reach \
	    from source to sink is " + std::to_string(flow_outcoming_source), 
	    true, false);
}
/**
 * @brief For given vertex v, find neighbor which the smallest height.  
 *
 * @tparam ValueT	Type of capacity/flow/excess
 * @tparam VertxeT	Type of vertex
 * 
 * @param[in] v		Index of vertex 
 * @param[in] n		Number of nodes
 * @param[in] height	Function of height on nodes 
 * @param[in] capacity	Function of capacity on edges
 * @param[in] flow	Function of flow on edges
 *
 * return Index the lowest neighbor of vertex v
 */
template<typename ValueT, typename VertexT>
VertexT find_lowest(VertexT v, VertexT n, VertexT* height, ValueT** capacity, 
	ValueT** flow){
    VertexT lowest = 2*n + 1;
    VertexT lowest_id = -1;
    for (VertexT i=0; i<n; ++i){
	if (capacity[v][i] - flow[v][i] > 0){
	    if (lowest_id == -1 || height[i] < lowest){
		lowest = height[i];
		lowest_id = i;
	    }
	}
    }
    return lowest_id;
}
/**
  * @brief Relabel: increases height of given vertex
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  *
  * @param[in] i	Index of vertex
  * @param[in] n	Number of nodes
  * @param[in] height	Function of height on nodes
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  *
  * return True if something changed, false otherwise
  */
template<typename ValueT, typename VertexT>
bool relabel(VertexT i, VertexT n, VertexT* height, ValueT* capacity,
	ValueT* flow){
    VertexT lowest_neighbour = find_lowest(i, n, height, capacity, flow);
    if (lowest_neighbour != -1 && height[lowest_neighbour] >= height[i]){
	height[i] = height[lowest_neighbour] + 1;
	return true;
    }
    return false;
}
/**
  * @brief Push: transfers flow from given vertex to neighbors in residual 
  *	   network which are lower than it.
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  *
  * @param[in] i	Index of vertex
  * @param[in] n	Number of nodes
  * @param[in] excess	Function of excess on nodes
  * @param[in] height	Function of height on nodes
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  *
  * return True if something changed, false otherwise
  */
template<typename ValueT, typename VertexT>
bool push(VertexT i, VertexT n, ValueT* excess, VertexT* height,
	ValueT** capacity, ValueT** flow){
    bool update = false;
    if (excess[i] > 0){
	for (int j=0; j<n; ++j){
	    if (capacity[i][j] - flow[i][j] > 0 && height[j] < height[i]){
		ValueT move = std::min(capacity[i][j]-flow[i][j], excess[i]);
		excess[i] -= move;
		excess[j] += move;
		flow[i][j] += move;
		flow[j][i] -= move;
		update = true;
	    }
	}
    }
    return update;
}
/**
  * @brief Push relabel reference algorithm
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  *
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  * @param[in] excess	Function of excess on nodes
  * @param[in] height	Function of height on nodes
  * @param[in] n	Number of nodes
  * @param[in] source	Source vertex
  * @param[in] sink	Sink vertex
  *
  * return Value of computed max flow
  */
template<typename ValueT, typename VertexT>
ValueT max_flow(ValueT** capacity, ValueT** flow, ValueT* excess, 
	VertexT* height, VertexT n, VertexT source, VertexT sink){
    bool update = true;
    while (update){
	update = false;
	for (VertexT i=0; i<n; ++i){
	    if (i == sink)
		continue;
	    if (excess[i] > 0){
		if (push(i, n, excess, height, capacity, flow) ||
			relabel(i, n, height, capacity, flow))
		    update = true;
	    }
	}
    }
    return excess[sink];
}

/****************************************************************************
 * MF Testing Routines
 ***************************************************************************/

/**
 * @brief Simple CPU-based reference MF implementations
 *
 * @tparam GraphT   Type of the graph
 * @tparam ValueT   Type of the capacity/flow/excess
 *
 * @param[in]  parameters Running parameters
 * @param[in]  graph      Input graph
 * @param[in]  src        The source vertex
 * @param[in]  sin        The sink vertex
 * @param[out] maxflow	  Value of computed maxflow reached sink
 * @param[out] edges_flow Computed flows on edges
 *
 * \return     double      Time taken for the MF
 */
template <typename VertexT, typename ValueT, typename GraphT>
double CPU_Reference(
	util::Parameters &parameters,
	GraphT &graph,
	VertexT src,
	VertexT sin,
	ValueT &maxflow,
	ValueT **edge_flow)
{

#ifdef BOOST_FOUND
    using namespace boost;

    //typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  GValueT;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list_traits<vecS, vecS, undirectedS> Traits;
    typedef adjacency_list<vecS, vecS, undirectedS, 
	    no_property,
            property <edge_capacity_t, ValueT,
	    property <edge_residual_capacity_t, ValueT,
	    property <edge_reverse_t, Traits::edge_descriptor> > > > BGraphT;
    typedef typename graph_traits<BGraphT>::vertex_descriptor 
							    vertex_descriptor;
    typedef typename graph_traits<BGraphT>::edge_descriptor edge_descriptor;
    std::vector<vertex_descriptor> verts;
    BGraphT g;

    for (VertexT v = 0; v < graph.nodes; ++v)
	verts.push_back(add_vertex(g));
    
    auto source = verts[src];
    auto sink = verts[sin];

    property_map<BGraphT, edge_capacity_t>::type capacity = 
	get(edge_capacity, g);

    property_map<BGraphT, edge_reverse_t>::type rev =
	get(edge_reverse, g);

    property_map<BGraphT, edge_residual_capacity_t>::type residual_capacity =
	get(edge_residual_capacity, g);

    for (VertexT e = 0; e < graph.edges; ++e){
	VertexT x = graph.edge_paris[e].x;
	VertexT y = graph.edge_paris[e].y;
	ValueT cap = graph.edge_values[e];
	edge_descriptor e1, e2;
	bool in1, in2;
	boost::tie(e1, in1) = add_edge(verts[x], verts[y], g);
	boost::tie(e2, in2) = add_edge(verts[y], verts[x], g);
	if (!in1 || !in2){
	    std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
	    return -1;
	}
	capacity[e1] = cap;
	capacity[e2] = 0;
	rev[e1] = e2;
	rev[e2] = e1;
    }
    
    //
    // Perform Boost reference
    //

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    maxflow = push_relabel_max_flow(graph, source, sink);

    cpu_timer.Stop();
    double elapsed = cpu_timer.ElapsedMillis();
    
    util::PrintMsg("CPU PushRelabel finished in " + std::to_string(elapsed) +
	    " msec. " + "Max flow is equal " + std::to_string(maxflow), true);

    //
    // Extracting results on CPU
    //

    graph_traits<BGraphT>::vertex_iterator u_it, u_end;
    graph_traits<BGraphT>::out_edge_iterator e_it, e_end;
    for (tie(u_it, u_end) = vertices(g), u_it != u_end; ++u_it){
    for (tie(e_it, e_end) = out_edges(*u_it, g); e_it != e_end; ++e_it){
	if (capacity[*e_it] > 0){
	    ValueT e_f = capacity[*e_it] - residual_capacity[*e_it];
	    VertexT target = target(*e_it, g);
	    util::PrintMsg("flow on edge " + std::to_string(*u_it) + "-" +
		    std::to_string(target) + " is " + std::to_string(e_f));
	    edge_flow[*u_it][target] = e_f;
	}
    }
    }

    return elapsed;

#else

    typedef typename GraphT::CooT     CooT;

    ValueT** capacity = (ValueT**)malloc(sizeof(ValueT*)*graph.edges);
    ValueT** flow =	(ValueT**)malloc(sizeof(ValueT*)*graph.edges);
    for (VertexT v = 0; v < graph.nodes; ++v){
	capacity[v] = (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
	flow[v]	    = (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
    }
    for (VertexT e = 0; e < graph.edges; ++e){
	VertexT x = graph.CooT::edge_pairs[e].x;
	VertexT y = graph.CooT::edge_pairs[e].y;
	capacity[x][y] = graph.CooT::edge_values[e];
    }

    ValueT*   excess =  (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
    VertexT*  height = (VertexT*)malloc(sizeof(VertexT)*graph.nodes);
    for (VertexT v = 0; v < graph.nodes; ++v){
	excess[v] = height[v] = 0;
    }
    height[src] = 2 * graph.nodes + 1;
    excess[sin] = std::numeric_limits<ValueT>::max();
    
    //
    // Perform simple max flow reference
    //
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    ValueT computed_flow = max_flow(capacity, flow, excess, height, 
	    graph.nodes, src, sin);
    cpu_timer.Stop();
    double elapsed = cpu_timer.ElapsedMillis();
    util::PrintMsg("CPU PushRelabel finished in " + std::to_string(elapsed) +
	    " msec. " + "Max flow is equal " + std::to_string(computed_flow), 
	    true);

    //
    // Extracting results on CPU
    //

    std::swap(flow, edge_flow);

    return elapsed;

#endif
}

/**
 * @brief Validation of MF results
 *
 * @tparam     GraphT	      Type of the graph
 * @tparam     ValueT	      Type of the distances
 *
 * @param[in]  parameters     Excution parameters
 * @param[in]  graph	      Input graph
 * @param[in]  source	      The source vertex
 * @param[in]  sink           The sink vertex
 * @param[in]  h_flow	      Computed flow on edges 
 * @param[in]  ref_flow	      Reference flow on edges
 * @param[in]  verbose	      Whether to output detail comparsions
 *
 * \return     int  Number of errors
 */
template <typename GraphT, typename ValueT, typename VertexT>
int Validate_Results(
	util::Parameters  &parameters,
        GraphT		  &graph,
    	VertexT		  source,
    	VertexT		  sink,
        ValueT		  **h_flow,
        ValueT		  **ref_flow = NULL,
	bool		  verbose = true)
{
    typedef typename GraphT::CooT    CooT;

    int num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // Verify the result
    if (ref_flow != NULL)
    {
	for (VertexT v = 0; v < graph.nodes; ++v)
	{
	    for (VertexT u = 0; u < graph.nodes; ++u)
	    {
		if (!util::isValid(ref_flow[v][u])){
		    ref_flow[v][u] = util::PreDefinedValues<ValueT>::MinValue;
		}
	    }
	}

	util::PrintMsg("Flow Validity: ", !quiet, false);

	VertexT errors_num = util::CompareResults(
		h_flow, ref_flow, graph.nodes, true, quiet);
	if (errors_num > 0)
	{
	    util::PrintMsg(std::to_string(errors_num) + " errors occurred.", 
		    !quiet);
	    num_errors += errors_num;
	}
    }
    else if (ref_flow == NULL)
    {
	util::PrintMsg("Flow Validity: ", !quiet, false);
	VertexT errors_num = 0;
	for (VertexT v = 0; v < graph.nodes; ++v)
	{
	    if (v == source || v == sink)
		continue;
	    ValueT flow_incoming_v = 0;
	    ValueT flow_outcoming_v = 0;
	    for (VertexT u = 0; u < graph.nodes; ++u)
	    {
		ValueT flow_uv = h_flow[u][v];
		if (util::isValid(flow_uv))
		    flow_incoming_v += flow_uv;
		ValueT flow_vu = h_flow[v][u];
		if (util::isValid(flow_vu))
		    flow_outcoming_v += flow_vu;
	    }
	    if (flow_incoming_v == flow_outcoming_v)
		continue;
	    errors_num ++;
	    if (errors_num > 1)
		continue;
	    util::PrintMsg("FAIL: v[" + std::to_string(v) + "]: " +
		    " incoming flow != outcoming flow: " + 
		    std::to_string(flow_incoming_v) + " != " +
		    std::to_string(flow_outcoming_v), !quiet);
	}
	if (errors_num > 0)
	{
	    util::PrintMsg(std::to_string(errors_num) + " errors occurred.", 
		    !quiet);
	    num_errors += errors_num;
	} else {
	    util::PrintMsg("PASS", !quiet);
	}
    }

    if (!quiet && verbose)
    {
	// Display Solution
	util::PrintMsg("Max Flow of the GPU result:");
	DisplaySolution(h_flow, source, graph.nodes);
	if (ref_flow != NULL)
	{
	    util::PrintMsg("Max Flow of the CPU results:");
	    DisplaySolution(ref_flow, source, graph.nodes);
	}
	util::PrintMsg("");
    }

    return num_errors;
}

} // namespace mf
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

