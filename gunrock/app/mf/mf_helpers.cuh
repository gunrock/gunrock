// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mf_helpers.cuh
 *
 * @brief Helper functions for MF algorithm.
 */

//#define debug_aml(a...) printf(a)
#define debug_aml(a...)

#pragma once

#define MF_EPSILON 0.000001
#define MF_EPSILON_VALIDATE 1e-3

#include <map>

namespace gunrock {
namespace app {
namespace mf {

template <typename ValueT>
__host__ __device__ bool almost_eql(ValueT A, ValueT B,
                                    ValueT maxRelDiff = MF_EPSILON) {
  if (fabs(A - B) < maxRelDiff) return true;
  return false;
}

template <typename GraphT, typename VertexT, typename ValueT>
int relabeling(GraphT graph, VertexT source, VertexT sink, VertexT* height,
               VertexT* reverse, ValueT* flow) {
  typedef typename GraphT::CsrT CsrT;

  bool mark[graph.nodes];
  for (VertexT x = 0; x < graph.nodes; ++x) mark[x] = false;
  // memset(mark, false, graph.nodes * sizeof(mark[0]));
  VertexT que[graph.nodes];
  int first = 0, last = 0;
  que[last++] = sink;
  mark[sink] = true;
  auto H = (VertexT)0;
  height[sink] = H;
  int changed = 0;

  while (first < last) {
    auto v = que[first++];
    auto e_start = graph.CsrT::GetNeighborListOffset(v);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(v);
    auto e_end = e_start + num_neighbors;
    ++H;
    for (auto e = e_start; e < e_end; ++e) {
      auto neighbor = graph.CsrT::GetEdgeDest(e);
      auto c = graph.CsrT::edge_values[reverse[e]];
      auto f = flow[reverse[e]];
      if (mark[neighbor] || almost_eql(c, f)) continue;
      if (height[neighbor] != H) changed++;

      height[neighbor] = H;
      mark[neighbor] = true;
      que[last++] = neighbor;
    }
  }
  height[source] = graph.nodes;
  return changed;
}

template <typename GraphT, typename VertexT,
          typename SizeT = typename GraphT::SizeT>
__host__ void InitReverse(GraphT& graph,
                          std::map<std::pair<VertexT, VertexT>, SizeT>& edge_id,
                          VertexT* reverse) {
  typedef typename GraphT::CsrT CsrT;

  for (auto u = 0; u < graph.nodes; ++u) {
    auto e_start = graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    for (auto e = e_start; e < e_end; ++e) {
      auto v = graph.CsrT::GetEdgeDest(e);
      auto f = edge_id[std::make_pair(v, u)];
      reverse[e] = f;
      reverse[f] = e;
    }
  }
}
template <typename GraphT, typename VertexT = typename GraphT::VertexT,
          typename SizeT = typename GraphT::SizeT>
__host__ void GetEdgesIds(
    GraphT& graph, std::map<std::pair<VertexT, VertexT>, SizeT>& edge_id) {
  typedef typename GraphT::CsrT CsrT;
  for (VertexT u = 0; u < graph.nodes; ++u) {
    auto f_start = graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors2 = graph.CsrT::GetNeighborListLength(u);
    auto f_end = f_start + num_neighbors2;
    for (auto f = f_start; f < f_end; ++f) {
      auto v = graph.CsrT::GetEdgeDest(f);
      edge_id[std::make_pair(u, v)] = f;
    }
  }
}
template <typename GraphT, typename VertexT = typename GraphT::VertexT,
          typename SizeT = typename GraphT::SizeT>
__host__ void CorrectCapacity(
    GraphT& undirected_graph, GraphT& directed_graph,
    std::map<std::pair<VertexT, VertexT>, SizeT>& d_edge_id) {
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::ValueT ValueT;

  // Correct capacity values on reverse edges
  for (auto u = 0; u < undirected_graph.nodes; ++u) {
    auto e_start = undirected_graph.CsrT::GetNeighborListOffset(u);
    auto num_neighbors = undirected_graph.CsrT::GetNeighborListLength(u);
    auto e_end = e_start + num_neighbors;
    debug_aml("vertex %d\nnumber of neighbors %d", u, num_neighbors);
    for (auto e = e_start; e < e_end; ++e) {
      undirected_graph.CsrT::edge_values[e] = (ValueT)0;
      auto v = undirected_graph.CsrT::GetEdgeDest(e);
      // Looking for edge u->v in directed graph
      auto f = d_edge_id[std::make_pair(u, v)];
      auto cap_f = directed_graph.CsrT::edge_values[f];
      if (cap_f > (ValueT)0) undirected_graph.CsrT::edge_values[e] = cap_f;
    }
  }
}

}  // namespace mf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
