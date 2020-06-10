/*
 * @brief Unit tests for dynamic graph operations
 * @file test_simple_advance.h
 */

#include <gunrock/graphio/graphio.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/frontier.cuh>
#include <algorithm>
#include "helper.h"

using namespace gunrock;

TEST(simpleAdvance, UsingCSRGraph) {
  // rng
  std::mt19937 rng(0);

  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = false;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto &ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph);

  ref_csr_graph.Move(util::HOST, util::DEVICE);

  // call advance
  VertexT advance_src = rng() % nodes;
  std::vector<VertexT> result_frontier;
  advanceTester("SIMPLE", ref_csr_graph, advance_src, result_frontier);

  // sort results
  auto advance_src_offset = ref_csr_graph.row_offsets[advance_src];
  auto advance_src_len =
      ref_csr_graph.row_offsets[advance_src + 1] - advance_src_offset;
  VertexT *expected_frontier =
      &ref_csr_graph.column_indices[advance_src_offset];

  EXPECT_EQ(result_frontier.size(), advance_src_len);

  std::sort(result_frontier.begin(), result_frontier.end());
  std::sort(expected_frontier, expected_frontier + advance_src_len);

  for (int i = 0; i < result_frontier.size(); i++)
    EXPECT_EQ(result_frontier[i], expected_frontier[i]);

  ref_graph.Release(util::HOST | util::DEVICE);
}

TEST(simpleAdvance, UsingDYNGraph) {
  // rng
  std::mt19937 rng(0);

  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = false;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto &ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph);

  // convert CSR to dynamic graph
  WeightedDYNGraphT dynamic_graph;
  auto &ref_dynamic_graph = dynamic_graph.dyn();
  ref_dynamic_graph.FromCsr(ref_csr_graph);

  // call advance
  VertexT advance_src = rng() % nodes;
  std::vector<VertexT> result_frontier;
  advanceTester("SIMPLE", ref_dynamic_graph, advance_src, result_frontier);

  // sort results
  auto advance_src_offset = ref_csr_graph.row_offsets[advance_src];
  auto advance_src_len =
      ref_csr_graph.row_offsets[advance_src + 1] - advance_src_offset;
  VertexT *expected_frontier =
      &ref_csr_graph.column_indices[advance_src_offset];

  EXPECT_EQ(result_frontier.size(), advance_src_len);

  std::sort(result_frontier.begin(), result_frontier.end());
  std::sort(expected_frontier, expected_frontier + advance_src_len);

  for (int i = 0; i < result_frontier.size(); i++)
    EXPECT_EQ(result_frontier[i], expected_frontier[i]);

  ref_graph.Release(util::HOST | util::DEVICE);
  dynamic_graph.Release(util::HOST | util::DEVICE);
}
