/*
 * @brief Unit tests for graph iterator
 * @file test_graph_iterator.h
 */

#include <gunrock/graphio/graphio.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/frontier.cuh>
#include <algorithm>
#include "helper.h"

using namespace gunrock;

TEST(dynamicGraphIterator, MultipleBuckets) {
  // rng
  std::mt19937 rng(0);

  // graph parameters
  SizeT nodes = 1000;
  SizeT edges_per_node = 30;  // make sure we have multiple buckets
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
  VertexT advance_src = 0;
  std::vector<VertexT> result_frontier;
  advanceTester("SIMPLE", ref_dynamic_graph, advance_src, result_frontier);

  // sort results
  auto advance_src_offset = ref_csr_graph.row_offsets[advance_src];
  auto advance_src_len =
      ref_csr_graph.row_offsets[advance_src + 1] - advance_src_offset;
  VertexT *expected_frontier =
      &ref_csr_graph.column_indices[advance_src_offset];

  GTEST_ASSERT_EQ(result_frontier.size(), advance_src_len);

  std::sort(result_frontier.begin(), result_frontier.end());
  std::sort(expected_frontier, expected_frontier + advance_src_len);

  for (int i = 0; i < result_frontier.size(); i++)
    EXPECT_EQ(result_frontier[i], expected_frontier[i]);

  dynamic_graph.Release(util::HOST | util::DEVICE);
  ref_graph.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraphIterator, SingleBucketMultipleChains) {
  // rng
  std::mt19937 rng(0);

  // graph parameters
  SizeT nodes = 10000;
  SizeT edges_per_node = 1;  // make sure we have one bucket
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

  // advance source
  VertexT advance_src = 0;

  // insert enough edges to make sure there is collision
  using PairT = uint2;
  SizeT batch_size = 10000;
  util::Array1D<SizeT, PairT> edges_batch;
  util::Array1D<SizeT, ValueT> edges_batch_values;
  edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
  edges_batch_values.Allocate(batch_size, util::HOST | util::DEVICE);
  for (auto e = 0; e < batch_size; e++) {
    VertexT edge_src = advance_src;
    VertexT edge_dst = (rng() % (nodes - 1)) + 1;
    edges_batch[e] = make_uint2(edge_src, edge_dst);
    edges_batch_values[e] = 1;
  }

  // move to GPU
  edges_batch.Move(util::HOST, util::DEVICE);
  edges_batch_values.Move(util::HOST, util::DEVICE);

  // insert the edges batch
  bool directed_batch = true;
  ref_dynamic_graph.InsertEdgesBatch(edges_batch, edges_batch_values,
                                     batch_size, directed_batch, util::DEVICE);

  // call advance
  std::vector<VertexT> result_frontier;
  advanceTester("SIMPLE", ref_dynamic_graph, advance_src, result_frontier);

  // build the expected frontier
  auto advance_src_offset = ref_csr_graph.row_offsets[advance_src];
  auto advance_src_len =
      ref_csr_graph.row_offsets[advance_src + 1] - advance_src_offset;
  VertexT *src_neighbor_ptr =
      (ref_csr_graph.column_indices.GetPointer(util::HOST)) +
      advance_src_offset;
  std::set<VertexT> expected_frontier;
  for (auto e = 0; e < advance_src_len; e++) {
    expected_frontier.insert(*src_neighbor_ptr);
    src_neighbor_ptr++;
  }
  for (auto e = 0; e < batch_size; e++) {
    if (edges_batch[e].y != advance_src)
      expected_frontier.insert(edges_batch[e].y);
  }

  GTEST_ASSERT_EQ(result_frontier.size(), expected_frontier.size());

  // sort results
  std::sort(result_frontier.begin(), result_frontier.end());

  SizeT i = 0;
  for (auto e_f : expected_frontier) {
    EXPECT_EQ(result_frontier[i++], e_f);
  }
  dynamic_graph.Release(util::HOST | util::DEVICE);
  ref_graph.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraphIterator, MultipleBucketMultipleChains) {
  // rng
  std::mt19937 rng(0);

  // graph parameters
  SizeT nodes = 10000;
  SizeT edges_per_node = 32;  // make sure we have multiple buckets
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

  // advance source
  VertexT advance_src = 0;

  // insert enough edges to make sure there is collision
  using PairT = uint2;
  SizeT batch_size = 10000;
  util::Array1D<SizeT, PairT> edges_batch;
  util::Array1D<SizeT, ValueT> edges_batch_values;
  edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
  edges_batch_values.Allocate(batch_size, util::HOST | util::DEVICE);
  for (auto e = 0; e < batch_size; e++) {
    VertexT edge_src = advance_src;
    VertexT edge_dst = (rng() % (nodes - 1)) + 1;
    edges_batch[e] = make_uint2(edge_src, edge_dst);
    edges_batch_values[e] = 1;
  }

  // move to GPU
  edges_batch.Move(util::HOST, util::DEVICE);
  edges_batch_values.Move(util::HOST, util::DEVICE);

  // insert the edges batch
  bool directed_batch = true;
  ref_dynamic_graph.InsertEdgesBatch(edges_batch, edges_batch_values,
                                     batch_size, directed_batch, util::DEVICE);

  // call advance
  std::vector<VertexT> result_frontier;
  advanceTester("SIMPLE", ref_dynamic_graph, advance_src, result_frontier);

  // build the expected frontier
  auto advance_src_offset = ref_csr_graph.row_offsets[advance_src];
  auto advance_src_len =
      ref_csr_graph.row_offsets[advance_src + 1] - advance_src_offset;
  VertexT *src_neighbor_ptr =
      (ref_csr_graph.column_indices.GetPointer(util::HOST)) +
      advance_src_offset;
  std::set<VertexT> expected_frontier;
  for (auto e = 0; e < advance_src_len; e++) {
    expected_frontier.insert(*src_neighbor_ptr);
    src_neighbor_ptr++;
  }
  for (auto e = 0; e < batch_size; e++) {
    if (edges_batch[e].y != advance_src)
      expected_frontier.insert(edges_batch[e].y);
  }

  GTEST_ASSERT_EQ(result_frontier.size(), expected_frontier.size());

  // sort results
  std::sort(result_frontier.begin(), result_frontier.end());

  SizeT i = 0;
  for (auto e_f : expected_frontier) {
    EXPECT_EQ(result_frontier[i++], e_f);
  }
  dynamic_graph.Release(util::HOST | util::DEVICE);
  ref_graph.Release(util::HOST | util::DEVICE);
}
