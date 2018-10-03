// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @fille
 * gtf_test.cu
 *
 * @brief Test related functions for Max Flow algorithm.
 */

#define debug_aml(a...)
//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a); \
    printf("\n");}

#pragma once

#ifdef BOOST_FOUND
    // Boost includes for CPU Push Relabel Max Flow reference algorithms
    #include <boost/config.hpp>
    #include <iostream>
    #include <string>
    #include <boost/graph/push_relabel_max_flow.hpp>
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/read_dimacs.hpp>
#endif

namespace gunrock {
namespace app {
namespace gtf {

/*****************************************************************************
 * Housekeeping Routines
 ****************************************************************************/

/**
 * @brief Displays the GTF result
 *
 * @tparam ValueT     Type of capacity/flow/excess
 * @tparam VertxeT    Type of vertex
 *
 * @param[in] h_flow  Flow calculated on edges
 * @param[in] source  Index of source vertex
 * @param[in] nodes   Number of nodes
 */
template<typename GraphT, typename ValueT, typename VertexT>
void DisplaySolution(GraphT graph, ValueT* h_flow, VertexT* reverse,
	VertexT sink, VertexT nodes)
{
    typedef typename GraphT::CsrT CsrT;
    typedef typename GraphT::SizeT SizeT;
    ValueT flow_incoming_sink = 0;
    SizeT e_start = graph.CsrT::GetNeighborListOffset(sink);
    SizeT num_neighbors = graph.CsrT::GetNeighborListLength(sink);
    SizeT e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e)
    {
	ValueT flow = h_flow[reverse[e]];
	if (util::isValid(flow))
	    flow_incoming_sink += flow;
    }

    util::PrintMsg("The maximum amount of flow that is feasible to reach \
	    from source to sink is " + std::to_string(flow_incoming_sink),
	    true, false);
}
/**
 * @brief For given vertex v, find neighbor which the smallest height.
 *
 * @tparam ValueT	Type of capacity/flow/excess
 * @tparam VertxeT	Type of vertex
 * @tparam GraphT	Type of graph
 * @param[in] graph	Graph
 * @param[in] x		Index of vertex
 * @param[in] height	Function of height on nodes
 * @param[in] capacity	Function of capacity on edges
 * @param[in] flow	Function of flow on edges
 *
 * return Index the lowest neighbor of vertex x
 */
template<typename ValueT, typename VertexT, typename GraphT>
VertexT find_lowest(GraphT graph, VertexT x, VertexT* height, ValueT* flow,
	VertexT source){
    typedef typename GraphT::CsrT CsrT;
    auto e_start = graph.CsrT::GetNeighborListOffset(x);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
    auto e_end = e_start + num_neighbors;
    VertexT lowest;
    VertexT lowest_id = -1; //graph.edges is unreachable value
    for (auto e = e_start; e < e_end; ++e)
    {
	if (graph.CsrT::edge_values[e] - flow[e] > (ValueT)0){
	    auto y = graph.CsrT::GetEdgeDest(e);
	    if (lowest_id == -1 || height[y] < lowest){
		lowest = height[y];
		lowest_id = e;
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
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] x	Index of vertex
  * @param[in] height	Function of height on nodes
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  *
  * return True if something changed, false otherwise
  */
template<typename ValueT, typename VertexT, typename GraphT>
bool relabel(GraphT graph, VertexT x, VertexT* height, ValueT* flow,
	VertexT source){
    typedef typename GraphT::CsrT CsrT;
    auto e = find_lowest(graph, x, height, flow, source);
    // graph.edges is unreachable value = there is no valid neighbour
    if (e != -1){
	VertexT y = graph.CsrT::GetEdgeDest(e);
	if (height[y] >= height[x]){
//	    printf("relabel %d H: %d->%d, res-cap %d-%d: %lf\n", x, height[x],
//		    height[y]+1, x, y, graph.CsrT::edge_values[e]-flow[e]);
	    height[x] = height[y] + 1;
	    return true;
	}
    }
    return false;
}
/**
  * @brief Push: transfers flow from given vertex to neighbors in residual
  *	   network which are lower than it.
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] x	Index of vertex
  * @param[in] excess	Function of excess on nodes
  * @param[in] height	Function of height on nodes
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  *
  * return True if something changed, false otherwise
  */
template<typename ValueT, typename VertexT, typename GraphT>
bool push(GraphT& graph, VertexT x, ValueT* excess, VertexT* height,
	ValueT* flow, VertexT* reverse){
    typedef typename GraphT::CsrT CsrT;
    if (excess[x] > (ValueT)0){
	auto e_start = graph.CsrT::GetNeighborListOffset(x);
	auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
	auto e_end = e_start + num_neighbors;
	for (auto e = e_start; e < e_end; ++e){
	    auto y = graph.CsrT::GetEdgeDest(e);
	    auto c = graph.CsrT::edge_values[e];
	    if (c - flow[e] > (ValueT) 0 and height[x] > height[y]){
		auto move = std::min(c - flow[e], excess[x]);
//		printf("push %lf from %d (H=%d) to %d (H=%d)\n",
//			move, x, height[x], y, height[y]);
		excess[x] -= move;
		excess[y] += move;
		flow[e] += move;
		flow[reverse[e]] -= move;
		return true;
	    }
	}
    }
    return false;
}

template<typename ValueT, typename VertexT, typename GraphT>
void dfs(ValueT** rGraph, VertexT s, bool visited[], const int V) {
    visited[s] = true;
    for (int i = 0; i < V; i++)
       if (abs(rGraph[s][i]) > 1e-6 && !visited[i])
           dfs(rGraph, i, visited, V);
}

/**
  * @brief Push relabel reference algorithm
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] capacity Function of capacity on edges
  * @param[in] flow	Function of flow on edges
  * @param[in] excess	Function of excess on nodes
  * @param[in] height	Function of height on nodes
  * @param[in] source	Source vertex
  * @param[in] sink	Sink vertex
  * @param[in] reverse	For given edge returns reverse one
  *
  * return Value of computed max flow
  */
template<typename ValueT, typename VertexT, typename GraphT>
ValueT max_flow(GraphT& graph, ValueT* flow,
	VertexT source, VertexT sink, VertexT* reverse){

    debug_aml("CPU_Reference start");
    typedef typename GraphT::SizeT SizeT;

    debug_aml("no boost");

    debug_aml("graph nodes %d, edges %d source %d sink %d source %d",
	    graph.nodes, graph.edges, source, sink);

    typedef typename GraphT::CsrT CsrT;

    ValueT*   excess =  (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
    VertexT*  height = (VertexT*)malloc(sizeof(VertexT)*graph.nodes);
    for (VertexT v = 0; v < graph.nodes; ++v){
	     excess[v] = (ValueT)0;
	     height[v] = 0;
    }
    height[source] = graph.nodes;

    for (SizeT e = 0; e < graph.edges; ++e){
	     flow[e] = (ValueT) 0;
    }

    auto e_start = graph.CsrT::GetNeighborListOffset(source);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(source);
    auto e_end = e_start + num_neighbors;

    ValueT preflow = (ValueT) 0;
    for (SizeT e = e_start; e < e_end; ++e)
    {
	     auto y = graph.CsrT::GetEdgeDest(e);
	     auto c = graph.CsrT::edge_values[e];
	     excess[y] += c;
	     flow[e] = c;
	     flow[reverse[e]] = -c;
	     preflow += c;
    }

    //
    // Perform simple max flow reference
    //
    debug_aml("perform simple max flow reference");
    debug_aml("source %d, sink %d", source, sink);
    debug_aml("source excess %lf, sink excess %lf", excess[source], excess[sink]);
    debug_aml("pre flow push from source %lf", preflow);
    debug_aml("source height %d, sink height %d", height[source], height[sink]);

    bool update = true;
    while (update){
	update = false;
	for (VertexT x = 0; x < graph.nodes; ++x){
	    if (x != sink and x != source and excess[x] > (ValueT)0){
		if (push(graph, x, excess, height, flow, reverse) or
			relabel(graph, x, height, flow, source))
		{
		    update = true;
		}
	    }
	}
    }
    /*
    for (auto u = source; u < graph.nodes; ++u)
    {
      auto e_start = graph.CsrT::GetNeighborListOffset(u);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
      auto e_end = e_start + num_neighbors;
      for (auto e = e_start; e < e_end; ++e)
      {
        auto v = graph.CsrT::GetEdgeDest(e);
        printf("flow(%d -> %d) = %f\n", u, v, flow[e]);
      }
    }

    GraphT u_graph = graph;
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
     bool *visited = new bool[graph.nodes];
     dfs(rGraph, source, visited, graph.nodes);
     for(int i = 0; i < graph.nodes; i++){
       cout << i << " " << visited[i] << endl;
     }
     */
    free(excess);
    free(height);
    return excess[sink];
}

/****************************************************************************
 * GTF Testing Routines
 ***************************************************************************/

/**
 * @brief Simple CPU-based reference GTF implementations
 *
 * @tparam GraphT   Type of the graph
 * @tparam VertexT  Type of the vertex
 * @tparam ValueT   Type of the capacity/flow/excess
 * @param[in]  parameters Running parameters
 * @param[in]  graph      Input graph
 * @param[in]  src        The source vertex
 * @param[in]  sin        The sink vertex
 * @param[out] maxflow	  Value of computed maxflow reached sink
 * @param[out] reverse	  Computed reverse
 * @param[out] edges_flow Computed flows on edges
 *
 * \return     double      Time taken for the GTF
 */
template <typename VertexT, typename ValueT, typename GraphT>
double CPU_Reference(
	util::Parameters &parameters,
	GraphT &graph,
	VertexT src,
	VertexT sin,
	ValueT &maxflow,
	VertexT *reverse,
	ValueT *flow)
{



    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    maxflow = max_flow(graph, flow, src, sin, reverse);

    cpu_timer.Stop();
    double elapsed = cpu_timer.ElapsedMillis();


    return elapsed;
}

/**
 * @brief Validation of GTF results
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
        ValueT		  *h_flow,
	VertexT		  *reverse,
        ValueT		  *ref_flow = NULL,
	bool		  verbose = true)
{
    typedef typename GraphT::CsrT   CsrT;
    typedef typename GraphT::SizeT  SizeT;

    int num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    auto nodes = graph.nodes;

    auto e_start = graph.CsrT::GetNeighborListOffset(sink);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(sink);
    auto e_end = e_start + num_neighbors;
    ValueT flow_incoming_sink = 0;
    for (auto e = e_start; e < e_end; ++e)
    {
	auto flow_e_in = h_flow[reverse[e]];
	if (util::isValid(flow_e_in))
	    flow_incoming_sink += flow_e_in;
    }
    util::PrintMsg("Max Flow GPU = " + std::to_string(flow_incoming_sink));


    // Verify the result
    if (ref_flow != NULL)
    {
	util::PrintMsg("Flow Validity (compare results):\n", !quiet, false);

	auto ref_flow_incoming_sink = (ValueT)0;
	for (auto e = e_start; e < e_end; ++e)
	{
	    auto flow_e_in = ref_flow[reverse[e]];
	    if (util::isValid(flow_e_in))
		ref_flow_incoming_sink += flow_e_in;
	}
	if (fabs(flow_incoming_sink-ref_flow_incoming_sink) > 1e-12)
	{
	    ++num_errors;
	    debug_aml("flow_incoming_sink %lf, ref_flow_incoming_sink %lf", \
		    flow_incoming_sink, ref_flow_incoming_sink);
	}

	if (num_errors > 0)
	{
	    util::PrintMsg(std::to_string(num_errors) + " errors occurred.",
		    !quiet);
	}else
	{
	    util::PrintMsg("PASS", !quiet);
	}
    }
    else if (ref_flow == NULL)
    {
	util::PrintMsg("Flow Validity:\n", !quiet, false);

	int errors_num = 0;
	for (VertexT v = 0; v < nodes; ++v)
	{
	    if (v == source || v == sink)
		continue;
	    auto e_start = graph.CsrT::GetNeighborListOffset(v);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
            auto e_end = e_start + num_neighbors;
	    ValueT flow_v = (ValueT) 0;
            for (auto e = e_start; e < e_end; ++e)
            {
		if (util::isValid(h_flow[e]))
		    flow_v += h_flow[e];
		else{
		    ++errors_num;
		    printf("flow for edge %d is invalid\n", e);
		}
	    }
	    if (fabs(flow_v) > 1e-12){
		printf("summary flow for vertex %d is %lf > %llf\n",
			v, fabs(flow_v), 1e-12);
	    }else
		continue;
	    ++errors_num;
	    util::PrintMsg("FAIL: for vertex " + std::to_string(v) +
		    " summary flow is " + std::to_string(flow_v) +
		    " not equal 0", !quiet);
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
	DisplaySolution(graph, h_flow, reverse, sink, graph.nodes);
	if (ref_flow != NULL)
	{
	    util::PrintMsg("Max Flow of the CPU results:");
	    DisplaySolution(graph, ref_flow, reverse, sink, graph.nodes);
	}
	util::PrintMsg("");
    }

    return num_errors;
}

} // namespace gtf
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
