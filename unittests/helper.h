/*
 * @brief Helper functions for unit tests
 * @file helper.h
 */
#pragma once

#include <gunrock/graphio/graphio.cuh>
#include <unordered_set>

using namespace gunrock;

using SizeT = uint32_t;
using VertexT = uint32_t;
using ValueT = uint32_t;

using WeightedCSRGraphT =
    app::TestGraph<VertexT, SizeT, ValueT,
                   graph::HAS_EDGE_VALUES | graph::HAS_CSR>;

using WeightedDYNGraphT =
    app::TestGraph<VertexT, SizeT, ValueT,
                   graph::HAS_EDGE_VALUES | graph::HAS_CSR | graph::HAS_DYN>;

using DynamicHostGraphT = std::vector<std::unordered_set<VertexT>>;

void RandomWeightedGraphToCsr(DynamicHostGraphT& ref_dyn,
                              WeightedCSRGraphT& ref_graph,
                              int const_weights = 0) {
  // rng
  std::mt19937 rng(0);

  auto ref_csr_graph = ref_graph.csr();

  // Populate the CSR with the adj. list
  SizeT cur_offset = 0;
  for (auto v = 0; v < ref_dyn.size(); v++) {
    auto& v_edges = ref_dyn[v];
    ref_csr_graph.row_offsets[v] = cur_offset;
    for (const auto& e : v_edges) {
      ref_csr_graph.column_indices[cur_offset] = e;
      if (const_weights == 0)
        ref_csr_graph.edge_values[cur_offset] = rng();
      else
        ref_csr_graph.edge_values[cur_offset] = const_weights;

      ++cur_offset;
    }
  }
  ref_csr_graph.row_offsets[ref_dyn.size()] = cur_offset;
}
void GenerateRandomWeightedGraph(DynamicHostGraphT& ref_dyn, SizeT nodes,
                                 SizeT edges, SizeT edges_per_node,
                                 bool undirected_graph) {
  // rng
  std::mt19937 rng(0);

  // generate a random reference  graph
  for (VertexT v = 0; v < nodes; v++) {
    auto& v_edges = ref_dyn[v];
    SizeT added_edges = 0;
    do {
      VertexT random_edge = rng() % nodes;
      if (random_edge != v) {
        auto res = v_edges.insert(random_edge);
        if (res.second) {
          if (undirected_graph) {
            ref_dyn[random_edge].insert(v);
          }
          added_edges++;
        }
      }
    } while (added_edges != edges_per_node);
  }
}
void CompareWeightedCSRs(WeightedCSRGraphT& ref_graph,
                         WeightedDYNGraphT& result_graph) {
  // sort both CSR graphs
  auto& result_csr_graph = result_graph.csr();
  auto& ref_csr_graph = ref_graph.csr();

  result_csr_graph.Sort();
  ref_csr_graph.Sort();

  SizeT nodes = result_csr_graph.nodes;

  // Compare
  EXPECT_EQ(ref_csr_graph.row_offsets[nodes],
            result_csr_graph.row_offsets[nodes]);

  for (auto v = 0; v < nodes; v++) {
    EXPECT_EQ(ref_csr_graph.row_offsets[v], result_csr_graph.row_offsets[v]);

    auto start_eid = ref_csr_graph.row_offsets[v];
    auto end_eid = ref_csr_graph.row_offsets[v + 1];

    for (auto eid = start_eid; eid < end_eid; eid++) {
      EXPECT_EQ(ref_csr_graph.column_indices[eid],
                result_csr_graph.column_indices[eid]);
      EXPECT_EQ(ref_csr_graph.edge_values[eid],
                result_csr_graph.edge_values[eid]);
    }
  }
}
