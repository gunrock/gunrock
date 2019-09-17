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
//#define debug_aml(a...) printf(a);

#define debug_aml_VK(a...)
//#define debug_amlVK(a...) printf(a)

#pragma once

#ifdef BOOST_FOUND
// Boost includes for CPU Push Relabel Max Flow reference algorithms
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <iostream>
#include <string>
#endif

#include <algorithm>
#include <queue>
#include <map>

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
template <typename GraphT, typename ValueT, typename VertexT>
void DisplaySolution(GraphT graph, ValueT* h_flow, VertexT* reverse,
                     VertexT sink, VertexT nodes) {
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::SizeT SizeT;
  ValueT flow_incoming_sink = 0;
  SizeT e_start = graph.CsrT::GetNeighborListOffset(sink);
  SizeT num_neighbors = graph.CsrT::GetNeighborListLength(sink);
  SizeT e_end = e_start + num_neighbors;
  for (auto e = e_start; e < e_end; ++e) {
    ValueT flow = h_flow[reverse[e]];
    if (util::isValid(flow)) flow_incoming_sink += flow;
  }

  util::PrintMsg(
      "The maximum amount of flow that is feasible to reach \
	    from source to sink is " +
          std::to_string(flow_incoming_sink),
      true, false);
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
template <typename ValueT, typename VertexT, typename GraphT>
ValueT max_flow(GraphT& graph, std::queue<VertexT>& que, ValueT* flow,
                ValueT* excess, VertexT* height, VertexT source, VertexT sink,
                VertexT* reverse) {
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::SizeT SizeT;
  bool update = true;
  debug_aml("que size %d\n", que.size());
  int iter = 0;
  for (; !que.empty(); que.pop()) {
    update = false;
    VertexT x = que.front();
    while (excess[x] > MF_EPSILON) {
      auto e_start = graph.CsrT::GetNeighborListOffset(x);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
      auto e_end = e_start + num_neighbors;
      VertexT lowest_h;
      SizeT lowest_id = util::PreDefinedValues<SizeT>::InvalidValue;
      VertexT lowest_y;
      for (auto e = e_start; e < e_end; ++e) {
        auto y = graph.CsrT::GetEdgeDest(e);
        auto c = graph.CsrT::edge_values[e];
        auto move = std::min(c - flow[e], excess[x]);
        if (move > MF_EPSILON &&
            (!util::isValid(lowest_id) || height[y] < lowest_h)) {
          lowest_h = height[y];
          lowest_id = e;
          lowest_y = y;
        }
      }
      if (height[x] > lowest_h) {
        // push
        auto c = graph.CsrT::edge_values[lowest_id];
        auto move = std::min(c - flow[lowest_id], excess[x]);
        excess[x] -= move;
        excess[lowest_y] += move;
        flow[lowest_id] += move;
        flow[reverse[lowest_id]] -= move;
        if (lowest_y != source and lowest_y != sink) que.push(lowest_y);
        debug_aml("push %lf, from %d to %d\n", move, x, lowest_y);
      } else {
        // relabel
        height[x] = lowest_h + 1;
        ++iter;
      }
      if (update && iter > 0 && iter % 100 == 0) {
        int up = relabeling(graph, source, sink, height, reverse, flow);
        if (up == 0) update = false;
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
void minCut(GraphT& graph, VertexT src, ValueT* flow, int* min_cut,
            bool* vertex_reachabilities, ValueT* residuals) {
  typedef typename GraphT::CsrT CsrT;
  memset(vertex_reachabilities, true,
         graph.nodes * sizeof(vertex_reachabilities[0]));
  std::queue<VertexT> que;
  que.push(src);
  min_cut[src] = 1;

  for (auto e = 0; e < graph.edges; e++) {
    residuals[e] = graph.CsrT::edge_values[e] - flow[e];
  }

  while (!que.empty()) {
    auto v = que.front();
    que.pop();

    auto e_start = graph.CsrT::GetNeighborListOffset(v);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      auto u = graph.CsrT::GetEdgeDest(e);
      if (vertex_reachabilities[u] and abs(residuals[e]) > MF_EPSILON) {
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
template <typename VertexT, typename ValueT, typename GraphT, typename SizeT>
double CPU_Reference(util::Parameters& parameters, GraphT& graph,
                     std::map<std::pair<VertexT, VertexT>, SizeT>& edge_id,
                     VertexT src, VertexT sin, ValueT& maxflow,
                     VertexT* reverse, ValueT* flow) {
  debug_aml("CPU_Reference start\n");
  typedef typename GraphT::CsrT CsrT;

  double elapsed = 0;

#if (BOOST_FOUND == 1)

  debug_aml("boost found\n");
  using namespace boost;

  // Prepare Boost Datatype and Data structure
  typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
  typedef adjacency_list<
      vecS, vecS, directedS,
      // property<vertex_name_t, std::string>,
      property<vertex_name_t, std::string,
               property<vertex_index_t, long,
                        property<vertex_color_t, boost::default_color_type,
                                 property<vertex_distance_t, long,
                                          property<vertex_predecessor_t,
                                                   Traits::edge_descriptor>>>>>,

      property<edge_capacity_t, ValueT,
               property<edge_residual_capacity_t, ValueT,
                        property<edge_reverse_t, Traits::edge_descriptor>>>>
      Graph;

  Graph boost_graph;

  typename property_map<Graph, edge_capacity_t>::type capacity =
      get(edge_capacity, boost_graph);

  typename property_map<Graph, edge_reverse_t>::type rev =
      get(edge_reverse, boost_graph);

  typename property_map<Graph, edge_residual_capacity_t>::type
      residual_capacity = get(edge_residual_capacity, boost_graph);

  std::vector<Traits::vertex_descriptor> verts;
  for (VertexT v = 0; v < graph.nodes; ++v)
    verts.push_back(add_vertex(boost_graph));

  Traits::vertex_descriptor source = verts[src];
  Traits::vertex_descriptor sink = verts[sin];
  debug_aml("src = %d, sin %d\n", source, sink);

  for (VertexT x = 0; x < graph.nodes; ++x) {
    auto e_start = graph.CsrT::GetNeighborListOffset(x);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(x);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      VertexT y = graph.CsrT::GetEdgeDest(e);
      ValueT cap = graph.CsrT::edge_values[e];
      Traits::edge_descriptor e1, e2;
      bool in1, in2;
      tie(e1, in1) = add_edge(verts[x], verts[y], boost_graph);
      tie(e2, in2) = add_edge(verts[y], verts[x], boost_graph);
      if (!in1 || !in2) {
        debug_aml("error\n");
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
  // maxflow = edmonds_karp_max_flow(boost_graph, source, sink);
  // maxflow = push_relabel_max_flow(boost_graph, source, sink);
  maxflow = boykov_kolmogorov_max_flow(boost_graph, source, sink);
  cpu_timer.Stop();
  elapsed = cpu_timer.ElapsedMillis();

  fprintf(stderr, "CPU Elapsed: %lf ms, cpu_reference result %lf\n", elapsed,
          maxflow);
  printf("CPU Elapsed: %lf ms, maxflow result %lf\n", elapsed, maxflow);

  //
  // Extracting results on CPU
  //

  typename graph_traits<Graph>::vertex_iterator u_it, u_end;
  typename graph_traits<Graph>::out_edge_iterator e_it, e_end;
  for (tie(u_it, u_end) = vertices(boost_graph); u_it != u_end; ++u_it) {
    for (tie(e_it, e_end) = out_edges(*u_it, boost_graph); e_it != e_end;
         ++e_it) {
      if (capacity[*e_it] > 0) {
        ValueT e_f = capacity[*e_it] - residual_capacity[*e_it];
        VertexT t = target(*e_it, boost_graph);
        flow[edge_id[std::make_pair(*u_it, t)]] = e_f;
      }
    }
  }

#else

  debug_aml("no boost\n");

  debug_aml("graph nodes %d, edges %d source %d sink %d src %d\n", graph.nodes,
            graph.edges, src, sin);

  ValueT* excess = (ValueT*)malloc(sizeof(ValueT) * graph.nodes);
  VertexT* height = (VertexT*)malloc(sizeof(VertexT) * graph.nodes);
  for (VertexT v = 0; v < graph.nodes; ++v) {
    excess[v] = (ValueT)0;
    height[v] = (VertexT)0;
  }
  height[src] = graph.nodes;

  for (SizeT e = 0; e < graph.edges; ++e) {
    flow[e] = (ValueT)0;
  }

  relabeling(graph, src, sin, height, reverse, flow);

  auto e_start = graph.CsrT::GetNeighborListOffset(src);
  auto num_neighbors = graph.CsrT::GetNeighborListLength(src);
  auto e_end = e_start + num_neighbors;

  std::queue<VertexT> que;

  ValueT preflow = (ValueT)0;
  for (SizeT e = e_start; e < e_end; ++e) {
    auto y = graph.CsrT::GetEdgeDest(e);
    auto c = graph.CsrT::edge_values[e];
    excess[y] += c;
    flow[e] = c;
    flow[reverse[e]] = -c;
    preflow += c;
    que.push(y);
  }

  //
  // Perform simple max flow reference
  //
  debug_aml("perform simple max flow reference\n");
  debug_aml("source %d, sink %d\n", src, sin);
  debug_aml("source excess %lf, sink excess %lf\n", excess[src], excess[sin]);
  debug_aml("pre flow push from source %lf\n", preflow);
  debug_aml("source height %d, sink height %d\n", height[src], height[sin]);

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  maxflow = max_flow(graph, que, flow, excess, height, src, sin, reverse);
  printf("maxflow result %lf\n", maxflow);

  cpu_timer.Stop();
  elapsed = cpu_timer.ElapsedMillis();

  // for (auto u = 0; u < graph.nodes; ++u){
  //     auto e_start = graph.CsrT::GetNeighborListOffset(u);
  //     auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
  //     auto e_end = e_start + num_neighbors;
  //     for (auto e = e_start; e < e_end; ++e){
  //         auto v = graph.CsrT::GetEdgeDest(e);
  //         auto f = flow[e];
  //         if (v == sin){
  //     	    printf("flow(%d->%d) = %lf (incoming sink CPU)\n", u, v, f);
  //         }
  //     }
  // }

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
 * @param[in]  reverse	      Reverse array on edges
 * @param[in]  min_cut	      Array on nodes, 0 - source set, 1 - sink set
 * @param[in]  ref_max_flow	  Value of max flow for reference solution
 * @param[in]  ref_flow	      Reference flow on edges
 * @param[in]  verbose	      Whether to output detail comparsions
 *
 * \return     int  Number of errors
 */
template <typename GraphT, typename ValueT, typename VertexT>
int Validate_Results(util::Parameters& parameters, GraphT& graph,
                     VertexT source, VertexT sink, ValueT* h_flow,
                     VertexT* reverse, int* min_cut, ValueT ref_max_flow,
                     ValueT* ref_flow = NULL, bool verbose = true) {
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::SizeT SizeT;

  int num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");
  auto nodes = graph.nodes;

  ValueT flow_incoming_sink = (ValueT)0;
  for (auto u = 0; u < graph.nodes; ++u) {
    auto e_start = graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      auto v = graph.CsrT::GetEdgeDest(e);
      if (v != sink) continue;
      auto flow_e_in = h_flow[e];
      // printf("flow(%d->%d) = %lf (incoming sink)\n", u, v, flow_e_in);
      if (util::isValid(flow_e_in)) flow_incoming_sink += flow_e_in;
    }
  }
  util::PrintMsg("Max Flow GPU = " + std::to_string(flow_incoming_sink));
  fprintf(stderr, "lockfree maxflow %lf\n", flow_incoming_sink);

  // Verify min cut h_flow
  ValueT mincut_flow = (ValueT)0;
  for (auto u = 0; u < graph.nodes; ++u) {
    if (min_cut[u] == 1) {
      auto e_start = graph.CsrT::GetNeighborListOffset(u);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
      auto e_end = e_start + num_neighbors;
      for (auto e = e_start; e < e_end; ++e) {
        auto v = graph.CsrT::GetEdgeDest(e);
        if (min_cut[v] == 0) {
          auto f = graph.CsrT::edge_values[e];
          mincut_flow += f;
        }
      }
    }
  }
  util::PrintMsg("MIN CUT flow = " + std::to_string(mincut_flow));
  if (fabs(mincut_flow - flow_incoming_sink) > MF_EPSILON_VALIDATE) {
    ++num_errors;
    util::PrintMsg("FAIL: Min cut " + std::to_string(mincut_flow) +
                       " and max flow " + std::to_string(flow_incoming_sink) +
                       " are not equal",
                   !quiet);
    fprintf(stderr, "FAIL: Min cut %lf and max flow %lf are not equal\n",
            mincut_flow, flow_incoming_sink);
  }

  // Verify the result
  if (!quick and ref_flow != NULL) {
    util::PrintMsg("Flow Validity (compare results):\n", !quiet, false);

    auto ref_flow_incoming_sink = (ValueT)0;
    for (auto u = 0; u < graph.nodes; ++u) {
      auto e_start = graph.CsrT::GetNeighborListOffset(u);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
      auto e_end = e_start + num_neighbors;
      for (auto e = e_start; e < e_end; ++e) {
        auto v = graph.CsrT::GetEdgeDest(e);
        if (v != sink) continue;
        auto flow_e_in = ref_flow[e];
        if (util::isValid(flow_e_in)) ref_flow_incoming_sink += flow_e_in;
      }
    }

    if (fabs(flow_incoming_sink - ref_flow_incoming_sink) >
        MF_EPSILON_VALIDATE) {
      ++num_errors;
      debug_aml("flow_incoming_sink %lf, ref_flow_incoming_sink %lf\n",
                flow_incoming_sink, ref_flow_incoming_sink);
    }

    if (num_errors > 0) {
      util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    } else {
      util::PrintMsg("PASS", !quiet);
      // fprintf(stderr, "PASS\n");
    }
  } else {
    util::PrintMsg("Flow Validity:\n", !quiet, false);

    if (util::isValid(ref_max_flow) and
        fabs(flow_incoming_sink - ref_max_flow) > MF_EPSILON_VALIDATE) {
      ++num_errors;
      debug_aml("flow_incoming_sink %lf, ref_max_flow %lf\n",
                flow_incoming_sink, ref_max_flow);
    }

    for (VertexT v = 0; v < nodes; ++v) {
      if (v == source || v == sink) continue;
      auto e_start = graph.CsrT::GetNeighborListOffset(v);
      auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
      auto e_end = e_start + num_neighbors;
      ValueT flow_v = (ValueT)0;
      for (auto e = e_start; e < e_end; ++e) {
        if (util::isValid(h_flow[e]))
          flow_v += h_flow[e];
        else {
          ++num_errors;
          debug_aml("flow for edge %d is invalid\n", e);
        }
      }
      if (fabs(flow_v) > MF_EPSILON_VALIDATE) {
        debug_aml("summary flow for vertex %d is %lf > %llf\n", v, fabs(flow_v),
                  MF_EPSILON);
      } else
        continue;
      ++num_errors;
      util::PrintMsg("FAIL: for vertex " + std::to_string(v) +
                         " summary flow " + std::to_string(flow_v) +
                         " is not equal 0",
                     !quiet);
      fprintf(stderr, "FAIL: for vertex %d summary flow %lf is not equal 0\n",
              v, flow_v);
    }
    if (num_errors > 0) {
      util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    } else {
      util::PrintMsg("PASS", !quiet);
      // fprintf(stderr, "PASS\n");
    }
  }

  if (!quick && verbose) {
    // Display Solution
    util::PrintMsg("Max Flow of the GPU result:");
    DisplaySolution(graph, h_flow, reverse, sink, graph.nodes);
    if (ref_flow != NULL) {
      util::PrintMsg("Max Flow of the CPU results:");
      DisplaySolution(graph, ref_flow, reverse, sink, graph.nodes);
    }
    util::PrintMsg("");
  }

  return num_errors;
}

}  // namespace mf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
