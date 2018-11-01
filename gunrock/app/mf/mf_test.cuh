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

#define debug_aml(a...)
//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a); \
    printf("\n");}

#pragma once

#ifdef BOOST_FOUND
    // Boost includes for CPU Push Relabel Max Flow reference algorithms
    #include <boost/config.hpp>
    #include <iostream>
    #include <string>
    #include <boost/graph/edmonds_karp_max_flow.hpp>
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/read_dimacs.hpp>
#endif

#include <queue>

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
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::CsrT CsrT;

    auto e_start = graph.CsrT::GetNeighborListOffset(x);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
    auto e_end = e_start + num_neighbors;
    VertexT lowest;
    SizeT lowest_id = util::PreDefinedValues<SizeT>::InvalidValue; 
    for (auto e = e_start; e < e_end; ++e)
    {
        //if (graph.CsrT::edge_values[e] - flow[e] > (ValueT)0){
        if (graph.CsrT::edge_values[e] - flow[e] > MF_EPSILON){
            auto y = graph.CsrT::GetEdgeDest(e);
            if (!util::isValid(lowest_id) || height[y] < lowest){
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
    if (util::isValid(e)) {
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
    //if (excess[x] > (ValueT)0){
    if (excess[x] > MF_EPSILON){
	auto e_start = graph.CsrT::GetNeighborListOffset(x);
	auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
	auto e_end = e_start + num_neighbors;
	for (auto e = e_start; e < e_end; ++e){
	    auto y = graph.CsrT::GetEdgeDest(e);
	    auto c = graph.CsrT::edge_values[e];
	    //if (c - flow[e] > (ValueT) 0 and height[x] > height[y]){
	    if (c - flow[e] > MF_EPSILON and height[x] > height[y]){
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
ValueT max_flow(GraphT& graph, ValueT* flow, ValueT* excess, VertexT* height, 
	VertexT source, VertexT sink, VertexT* reverse){
    bool update = true;

    int iter = 0;
    while (update) {
        ++iter;
        update = false;
        for (VertexT x = 0; x < graph.nodes; ++x){
            //if (x != sink and x != source and excess[x] > (ValueT)0){
            if (x != sink and x != source and excess[x] > MF_EPSILON){
                if (push(graph, x, excess, height, flow, reverse) or 
                        relabel(graph, x, height, flow, source))
                {
                    update = true;
                    if (iter > 0 && iter % 100 == 0)
                        relabeling(graph, source, sink, height, reverse, flow); 
                }
            }
        }
    }

    return excess[sink];
}

/**
  * @brief Min Cut algorithm
  *
  * @tparam ValueT	Type of capacity/flow/excess
  * @tparam VertxeT	Type of vertex
  * @tparam GraphT	Type of graph
  * @param[in] graph	Graph
  * @param[in] source	Source vertex
  * @param[in] sink	Sink vertex
  * @param[in] flow	Function of flow on edges
  * @param[out] min_cut	Function on nodes, 1 = connected to source, 0 = sink
  *
  */
template <typename VertexT, typename ValueT, typename GraphT>
void minCut(GraphT& graph, VertexT  src, ValueT* flow, int* min_cut, 
	    bool* vertex_reachabilities, ValueT* residuals)
{
    typedef typename GraphT::CsrT CsrT;
    memset(vertex_reachabilities, true, graph.nodes * sizeof(vertex_reachabilities[0]));
    std::queue<VertexT> que;
    que.push(src);
    min_cut[src] = 1;

    for (auto e = 0; e < graph.edges; e++) {
	residuals[e] = graph.CsrT::edge_values[e] - flow[e];
    }

    while (! que.empty()){
        auto v = que.front(); que.pop();

        auto e_start = graph.CsrT::GetNeighborListOffset(v);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            auto u = graph.CsrT::GetEdgeDest(e);
            if (vertex_reachabilities[u] and abs(residuals[e]) > MF_EPSILON){
                vertex_reachabilities[u] = false;
                que.push(u);
                min_cut[u] = 1;
            }
        }
    }
}

/****************************************************************************
 * MF Testing Routines
 ***************************************************************************/

/**
 * @brief Simple CPU-based reference MF implementations
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
 * \return     double      Time taken for the MF
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

    debug_aml("CPU_Reference start");
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::CsrT CsrT;

    double elapsed = 0;

#if (BOOST_FOUND==1)
    
    debug_aml("boost found");
    using namespace boost;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
    typedef adjacency_list < vecS, vecS, directedS, 
	    property < vertex_name_t, std::string >,
	    property < edge_capacity_t, ValueT,
	    property < edge_residual_capacity_t, ValueT,
	    property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;
    
    Graph boost_graph;

    typename property_map < Graph, edge_capacity_t >::type 
	capacity = get(edge_capacity, boost_graph);

    typename property_map < Graph, edge_reverse_t >::type 
	rev = get(edge_reverse, boost_graph);

    typename property_map < Graph, edge_residual_capacity_t >::type 
	residual_capacity = get(edge_residual_capacity, boost_graph);

    std::vector<Traits::vertex_descriptor> verts;
    for (VertexT v = 0; v < graph.nodes; ++v)
	verts.push_back(add_vertex(boost_graph));
    
    Traits::vertex_descriptor source = verts[src];
    Traits::vertex_descriptor sink = verts[sin];
    debug_aml("src = %d, sin %d", source, sink);

    for (VertexT x = 0; x < graph.nodes; ++x){
	auto e_start = graph.CsrT::GetNeighborListOffset(x);
	auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
	auto e_end = e_start + num_neighbors;
	for (auto e = e_start; e < e_end; ++e){
	    VertexT y = graph.CsrT::GetEdgeDest(e);
	    ValueT cap = graph.CsrT::edge_values[e];
	    if (fabs(cap) <= 1e-12)
		continue;
	    Traits::edge_descriptor e1, e2;
	    bool in1, in2;
	    tie(e1, in1) = add_edge(verts[x], verts[y], boost_graph);
	    tie(e2, in2) = add_edge(verts[y], verts[x], boost_graph);
	    if (!in1 || !in2){
		debug_aml("error");
		return -1;
	    }
	    capacity[e1] = cap;
	    capacity[e2] = 0;
	    rev[e1] = e2;
	    rev[e2] = e1;
	}
    }
  
    //
    // Perform Boost reference
    //

    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    maxflow = edmonds_karp_max_flow(boost_graph, source, sink); 
    cpu_timer.Stop();
    elapsed = cpu_timer.ElapsedMillis();

    //
    // Extracting results on CPU
    //

    std::vector<std::vector<ValueT>> boost_flow; 
    boost_flow.resize(graph.nodes);
    for (auto x = 0; x < graph.nodes; ++x) 
	boost_flow[x].resize(graph.nodes, 0.0);
    typename graph_traits<Graph>::vertex_iterator u_it, u_end;
    typename graph_traits<Graph>::out_edge_iterator e_it, e_end;
    for (tie(u_it, u_end) = vertices(boost_graph); u_it != u_end; ++u_it){
	for (tie(e_it, e_end) = out_edges(*u_it, boost_graph); e_it != e_end; 
		++e_it){
	    if (capacity[*e_it] > 0){
		ValueT e_f = capacity[*e_it] - residual_capacity[*e_it];
	    	VertexT t = target(*e_it, boost_graph);
	    	//debug_aml("flow on edge %d - %d = %lf", *u_it, t, e_f);
	    	boost_flow[*u_it][t] = e_f;
	    }
	}
    }
    for (auto x = 0; x < graph.nodes; ++x){
	auto e_start = graph.CsrT::GetNeighborListOffset(x);
	auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
	auto e_end = e_start + num_neighbors;
	for (auto e = e_start; e < e_end; ++e){
	    VertexT y = graph.CsrT::GetEdgeDest(e);
	    flow[e] = boost_flow[x][y];
	}
    }

#else

    debug_aml("no boost");

    debug_aml("graph nodes %d, edges %d source %d sink %d src %d", 
	    graph.nodes, graph.edges, src, sin);

    ValueT*   excess =  (ValueT*)malloc(sizeof(ValueT)*graph.nodes);
    VertexT*  height = (VertexT*)malloc(sizeof(VertexT)*graph.nodes);
    for (VertexT v = 0; v < graph.nodes; ++v){
        excess[v] = (ValueT)0;
        height[v] = (VertexT)0;
    }
     
    for (SizeT e = 0; e < graph.edges; ++e){
        flow[e] = (ValueT) 0;
    }

    debug_aml("before relabeling");
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("height[%d] = %d", v, height[v]);
    }
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("excess[%d] = %lf", v, excess[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("flow[%d] = %lf", v, flow[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("capacity[%d] = %lf", v, graph.CsrT::edge_values[v]);
    }
    relabeling(graph, src, sin, height, reverse, flow);
    debug_aml("after relabeling");
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("height[%d] = %d", v, height[v]);
    }
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("excess[%d] = %lf", v, excess[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("flow[%d] = %lf", v, flow[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("capacity[%d] = %lf", v, graph.CsrT::edge_values[v]);
    }
    auto e_start = graph.CsrT::GetNeighborListOffset(src);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(src);
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

    debug_aml("after preflow");
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("height[%d] = %d", v, height[v]);
    }
    for (SizeT v = 0; v < graph.nodes; ++v){
        debug_aml("excess[%d] = %lf", v, excess[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("flow[%d] = %lf", v, flow[v]);
    }
    for (SizeT v = 0; v < graph.edges; ++v){
        debug_aml("capacity[%d] = %lf", v, graph.CsrT::edge_values[v]);
    }

    {
        auto e_start = graph.CsrT::GetNeighborListOffset(src);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(src);
        auto e_end = e_start + num_neighbors;
        for (SizeT e = e_start; e < e_end; ++e)
        {
            auto y = graph.CsrT::GetEdgeDest(e);
            debug_aml("height[%d] = %d", y, height[y]);
        }
        for (int i=0; i<graph.nodes; ++i){
            debug_aml("excess[%d] = %lf\n", i, excess[i]);
        }
    }

    //
    // Perform simple max flow reference
    //
    debug_aml("perform simple max flow reference");
    debug_aml("source %d, sink %d", src, sin);
    debug_aml("source excess %lf, sink excess %lf", excess[src], excess[sin]);
    debug_aml("pre flow push from source %lf", preflow);
    debug_aml("source height %d, sink height %d", height[src], height[sin]);
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    maxflow = max_flow(graph, flow, excess, height, src, sin, reverse);

//    for (auto u = 0; u < graph.nodes; ++u){
//	auto e_start = graph.CsrT::GetNeighborListOffset(u);
//	auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
//	auto e_end = e_start + num_neighbors;
//	for (auto e = e_start; e < e_end; ++e){
//	    auto v = graph.CsrT::GetEdgeDest(e);
//	    auto f = flow[e];
//	    if (v == sin){
//		printf("flow(%d->%d) = %lf (incoming sink CPU)\n", u, v, f);
//	    }
//	}
//    }
    
    cpu_timer.Stop();
    elapsed = cpu_timer.ElapsedMillis();

    free(excess);
    free(height);

#endif
    
    return elapsed;
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
        ValueT		  *h_flow,
        VertexT		  *reverse,
        int               *min_cut,
        ValueT		  *ref_flow = NULL,
        bool		  verbose = true)
{
    typedef typename GraphT::CsrT   CsrT;
    typedef typename GraphT::SizeT  SizeT;  

    int num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    bool quick = parameters.Get<bool>("quick");
    auto nodes = graph.nodes;

    ValueT flow_incoming_sink = (ValueT)0;
    for (auto u = 0; u < graph.nodes; ++u){
        auto e_start = graph.CsrT::GetNeighborListOffset(u);
        auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
        auto e_end = e_start + num_neighbors;
        for (auto e = e_start; e < e_end; ++e){
            auto v = graph.CsrT::GetEdgeDest(e);
            if (v != sink)
                continue;
            auto flow_e_in = h_flow[e];
            //printf("flow(%d->%d) = %lf (incoming sink)\n", u, v, flow_e_in);
            if (util::isValid(flow_e_in))
                flow_incoming_sink += flow_e_in;
        }
    }
    util::PrintMsg("Max Flow GPU = " + std::to_string(flow_incoming_sink));

    // Verify min cut h_flow
    ValueT mincut_flow = (ValueT)0;
    for (auto u = 0; u < graph.nodes; ++u){
        if (min_cut[u] == 1){
            auto e_start = graph.CsrT::GetNeighborListOffset(u);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
            auto e_end = e_start + num_neighbors;
            for (auto e = e_start; e < e_end; ++e){
                auto v = graph.CsrT::GetEdgeDest(e);
                if (min_cut[v] == 0){
                    auto f = graph.CsrT::edge_values[e];
                    mincut_flow += f;
                }
            }
        }
    }
    util::PrintMsg("MIN CUT flow = " + std::to_string(mincut_flow));
    if (fabs(mincut_flow - flow_incoming_sink) > MF_EPSILON_VALIDATE)
    {
        ++num_errors;
        util::PrintMsg("FAIL: Min cut " + std::to_string(mincut_flow) +
                " and max flow " + std::to_string(flow_incoming_sink) + 
                " are not equal", !quiet);
    }

    // Verify the result
    if (!quick)
    {
        util::PrintMsg("Flow Validity (compare results):\n", !quiet, false);

        auto ref_flow_incoming_sink = (ValueT)0;
        for (auto u = 0; u < graph.nodes; ++u){
            auto e_start = graph.CsrT::GetNeighborListOffset(u);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
            auto e_end = e_start + num_neighbors;
            for (auto e = e_start; e < e_end; ++e){
                auto v = graph.CsrT::GetEdgeDest(e);
                if (v != sink)
                    continue;
                auto flow_e_in = ref_flow[e];
                if (util::isValid(flow_e_in))
                    ref_flow_incoming_sink += flow_e_in;
            }
        }

        if (fabs(flow_incoming_sink-ref_flow_incoming_sink) > MF_EPSILON_VALIDATE)
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
    else
    {
        util::PrintMsg("Flow Validity:\n", !quiet, false);

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
                    ++num_errors;
                    debug_aml("flow for edge %d is invalid\n", e);
                }
            }
            if (fabs(flow_v) > MF_EPSILON_VALIDATE){
                debug_aml("summary flow for vertex %d is %lf > %llf\n", 
                        v, fabs(flow_v), 1e-12);
            }else
                continue;
            ++num_errors;
            util::PrintMsg("FAIL: for vertex " + std::to_string(v) +
                    " summary flow " + std::to_string(flow_v) + 
                    " is not equal 0", !quiet);
        }
        if (num_errors > 0)
        {
            util::PrintMsg(std::to_string(num_errors) + " errors occurred.", 
                    !quiet);
        } else {
            util::PrintMsg("PASS", !quiet);
        }
    }

    if (!quick && verbose)
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

} // namespace mf
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

