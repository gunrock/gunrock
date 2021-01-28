/*
 * @brief Unit tests for dynamic graph operations
 * @file test_dynamic_graph.h
 */

#include <gunrock/graphio/graphio.cuh>
#include "helper.h"

using namespace gunrock;

TEST(dynamicGraph, buildDirectedWeighted) {
  // rng
  std::mt19937 rng(0);

  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node;
  bool directed = true;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto& ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph);

  // convert CSR to dynamic graph
  WeightedDYNGraphT result_graph;
  auto& result_dynamic_graph = result_graph.dyn();
  result_dynamic_graph.FromCsr(ref_csr_graph);

  // convert dynamic graph back to CSR
  auto& result_csr_graph = result_graph.csr();
  result_graph.csr().Allocate(nodes, edges, util::HOST | util::DEVICE);
  result_dynamic_graph.ToCsr(result_csr_graph);
  result_csr_graph.Move(util::DEVICE, util::HOST);

  CompareWeightedCSRs(ref_graph, result_graph);

  result_dynamic_graph.Release();
  ref_graph.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraph, buildUndirectedWeighted) {
  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = false;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto& ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph);

  // convert CSR to dynamic graph
  WeightedDYNGraphT result_graph;
  auto& result_dynamic_graph = result_graph.dyn();
  result_dynamic_graph.FromCsr(ref_csr_graph);

  // convert dynamic graph back to CSR
  auto& result_csr_graph = result_graph.csr();
  result_csr_graph.Allocate(nodes, edges, util::HOST | util::DEVICE);
  result_dynamic_graph.ToCsr(result_csr_graph);
  result_csr_graph.Move(util::DEVICE, util::HOST);

  CompareWeightedCSRs(ref_graph, result_graph);

  result_dynamic_graph.Release();
  ref_graph.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraph, insertUndirectedWeighted) {
  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = false;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto& ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph, 1);

  // convert CSR to dynamic graph
  WeightedDYNGraphT result_graph;
  auto& result_dynamic_graph = result_graph.dyn();
  result_dynamic_graph.FromCsr(ref_csr_graph);

  // generate a random batch of edges to insert
  using PairT = uint2;
  SizeT batch_size = 10000;
  util::Array1D<SizeT, PairT> edges_batch;
  util::Array1D<SizeT, ValueT> edges_batch_values;
  edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
  edges_batch_values.Allocate(batch_size, util::HOST | util::DEVICE);
  std::mt19937 rng(0);

  for (auto e = 0; e < batch_size; e++) {
    VertexT edge_src = rng() % nodes;
    VertexT edge_dst = rng() % nodes;
    edges_batch[e] = make_uint2(edge_src, edge_dst);
    edges_batch_values[e] = 1;
  }

  // move to GPU
  edges_batch.Move(util::HOST, util::DEVICE);
  edges_batch_values.Move(util::HOST, util::DEVICE);

  // insert the edges batch
  bool directed_batch = true;
  result_dynamic_graph.InsertEdgesBatch(edges_batch, edges_batch_values,
                                        batch_size, directed_batch,
                                        util::DEVICE);

  // Apply batch to host graph & generate values as well
  SizeT new_edges_count = edges;
  for (auto e = 0; e < batch_size; e++) {
    if (edges_batch[e].x != edges_batch[e].y) {
      auto res_0 = ref_dyn[edges_batch[e].x].insert(edges_batch[e].y);
      auto res_1 = ref_dyn[edges_batch[e].y].insert(edges_batch[e].x);
      if (res_0.second) new_edges_count++;
      if (res_1.second) new_edges_count++;
    }
  }

  // New static graph
  WeightedCSRGraphT ref_graph_updated;
  auto& ref_csr_graph_updated = ref_graph_updated.csr();
  ref_csr_graph_updated.Allocate(nodes, new_edges_count, util::HOST);
  ref_csr_graph_updated.directed = directed;
  RandomWeightedGraphToCsr(ref_dyn, ref_graph_updated, 1);

  // convert dynamic graph back to CSR
  auto& result_csr_graph = result_graph.csr();
  result_csr_graph.Allocate(nodes, new_edges_count, util::HOST | util::DEVICE);
  result_dynamic_graph.ToCsr(result_csr_graph);
  result_csr_graph.Move(util::DEVICE, util::HOST);

  CompareWeightedCSRs(ref_graph_updated, result_graph);

  result_dynamic_graph.Release();
  ref_graph.Release(util::HOST | util::DEVICE);
  ref_graph_updated.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraph, insertDirectedWeighted) {
  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = true;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto& ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph, 1);

  // convert CSR to dynamic graph
  WeightedDYNGraphT result_graph;
  auto& result_dynamic_graph = result_graph.dyn();
  result_dynamic_graph.FromCsr(ref_csr_graph);

  // generate a random batch of edges to insert
  using PairT = uint2;
  SizeT batch_size = 10000;
  util::Array1D<SizeT, PairT> edges_batch;
  util::Array1D<SizeT, ValueT> edges_batch_values;
  edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
  edges_batch_values.Allocate(batch_size, util::HOST | util::DEVICE);
  std::mt19937 rng(0);

  for (auto e = 0; e < batch_size; e++) {
    VertexT edge_src = rng() % nodes;
    VertexT edge_dst = rng() % nodes;
    edges_batch[e] = make_uint2(edge_src, edge_dst);
    edges_batch_values[e] = 1;
  }

  // move to GPU
  edges_batch.Move(util::HOST, util::DEVICE);
  edges_batch_values.Move(util::HOST, util::DEVICE);

  // insert the edges batch
  bool directed_batch = true;
  result_dynamic_graph.InsertEdgesBatch(edges_batch, edges_batch_values,
                                        batch_size, directed_batch,
                                        util::DEVICE);

  // Apply batch to host graph & generate values as well
  SizeT new_edges_count = edges;
  for (auto e = 0; e < batch_size; e++) {
    if (edges_batch[e].x != edges_batch[e].y) {
      auto res_0 = ref_dyn[edges_batch[e].x].insert(edges_batch[e].y);
      if (res_0.second) new_edges_count++;
    }
  }

  // New static graph
  WeightedCSRGraphT ref_graph_updated;
  auto& ref_csr_graph_updated = ref_graph_updated.csr();
  ref_csr_graph_updated.Allocate(nodes, new_edges_count, util::HOST);
  ref_csr_graph_updated.directed = directed;
  RandomWeightedGraphToCsr(ref_dyn, ref_graph_updated, 1);

  // convert dynamic graph back to CSR
  auto& result_csr_graph = result_graph.csr();
  result_csr_graph.Allocate(nodes, new_edges_count, util::HOST | util::DEVICE);
  result_dynamic_graph.ToCsr(result_csr_graph);
  result_csr_graph.Move(util::DEVICE, util::HOST);

  CompareWeightedCSRs(ref_graph_updated, result_graph);

  result_dynamic_graph.Release();
  ref_graph.Release(util::HOST | util::DEVICE);
  ref_graph_updated.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraph, deleteUndirectedWeighted) {
  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = false;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto& ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph, 1);

  // convert CSR to dynamic graph
  WeightedDYNGraphT result_graph;
  auto& result_dynamic_graph = result_graph.dyn();
  result_dynamic_graph.FromCsr(ref_csr_graph);

  // generate a random batch of edges to insert
  using PairT = uint2;
  SizeT batch_size = 20;
  util::Array1D<SizeT, PairT> edges_batch;
  edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
  std::mt19937 rng(0);

  for (auto e = 0; e < batch_size; e++) {
    VertexT edge_src = rng() % nodes;
    VertexT edge_dst = rng() % nodes;
    edges_batch[e] = make_uint2(edge_src, edge_dst);
  }

  // move the edges batch to the GPU
  edges_batch.Move(util::HOST, util::DEVICE);

  // delete the edges batch
  result_dynamic_graph.DeleteEdgesBatch(edges_batch, batch_size, util::DEVICE);

  // Apply batch to host graph & generate values as well
  SizeT new_edges_count = edges;
  for (auto e = 0; e < batch_size; e++) {
    if (edges_batch[e].x != edges_batch[e].y) {
      auto res_0 = ref_dyn[edges_batch[e].x].erase(edges_batch[e].y);
      auto res_1 = ref_dyn[edges_batch[e].y].erase(edges_batch[e].x);
      if (res_0) {
        new_edges_count--;
      }
      if (res_1) {
        new_edges_count--;
      }
    }
  }

  // New static graph
  WeightedCSRGraphT ref_graph_updated;
  auto& ref_csr_graph_updated = ref_graph_updated.csr();
  ref_csr_graph_updated.Allocate(nodes, new_edges_count, util::HOST);
  ref_csr_graph_updated.directed = directed;
  RandomWeightedGraphToCsr(ref_dyn, ref_graph_updated, 1);

  // convert dynamic graph back to CSR
  auto& result_csr_graph = result_graph.csr();
  result_csr_graph.Allocate(nodes, new_edges_count, util::HOST | util::DEVICE);
  result_dynamic_graph.ToCsr(result_csr_graph);
  result_csr_graph.Move(util::DEVICE, util::HOST);

  CompareWeightedCSRs(ref_graph_updated, result_graph);

  result_dynamic_graph.Release();
  ref_graph.Release(util::HOST | util::DEVICE);
  ref_graph_updated.Release(util::HOST | util::DEVICE);
}

TEST(dynamicGraph, deleteDirectedWeighted) {
  // graph parameters
  SizeT nodes = 100;
  SizeT edges_per_node = 5;
  SizeT edges = nodes * edges_per_node * 2;
  bool directed = true;

  // reference CSR graph
  WeightedCSRGraphT ref_graph;
  auto& ref_csr_graph = ref_graph.csr();
  ref_csr_graph.Allocate(nodes, edges, util::HOST);
  ref_csr_graph.directed = directed;

  DynamicHostGraphT ref_dyn(nodes);
  GenerateRandomWeightedGraph(ref_dyn, nodes, edges, edges_per_node, !directed);
  RandomWeightedGraphToCsr(ref_dyn, ref_graph, 1);

  // convert CSR to dynamic graph
  WeightedDYNGraphT result_graph;
  auto& result_dynamic_graph = result_graph.dyn();
  result_dynamic_graph.FromCsr(ref_csr_graph);

  // generate a random batch of edges to insert
  using PairT = uint2;
  SizeT batch_size = 20;
  util::Array1D<SizeT, PairT> edges_batch;
  edges_batch.Allocate(batch_size, util::HOST | util::DEVICE);
  std::mt19937 rng(0);

  for (auto e = 0; e < batch_size; e++) {
    VertexT edge_src = rng() % nodes;
    VertexT edge_dst = rng() % nodes;
    edges_batch[e] = make_uint2(edge_src, edge_dst);
  }

  // move the edges batch to the GPU
  edges_batch.Move(util::HOST, util::DEVICE);

  // delete the edges batch
  result_dynamic_graph.DeleteEdgesBatch(edges_batch, batch_size, util::DEVICE);

  // Apply batch to host graph & generate values as well
  SizeT new_edges_count = edges;
  for (auto e = 0; e < batch_size; e++) {
    if (edges_batch[e].x != edges_batch[e].y) {
      auto res_0 = ref_dyn[edges_batch[e].x].erase(edges_batch[e].y);
      if (res_0) {
        new_edges_count--;
      }
    }
  }

  // New static graph
  WeightedCSRGraphT ref_graph_updated;
  auto& ref_csr_graph_updated = ref_graph_updated.csr();
  ref_csr_graph_updated.Allocate(nodes, new_edges_count, util::HOST);
  ref_csr_graph_updated.directed = directed;
  RandomWeightedGraphToCsr(ref_dyn, ref_graph_updated, 1);

  // convert dynamic graph back to CSR
  auto& result_csr_graph = result_graph.csr();
  result_csr_graph.Allocate(nodes, new_edges_count, util::HOST | util::DEVICE);
  result_dynamic_graph.ToCsr(result_csr_graph);
  result_csr_graph.Move(util::DEVICE, util::HOST);

  CompareWeightedCSRs(ref_graph_updated, result_graph);

  result_dynamic_graph.Release();
  ref_graph.Release(util::HOST | util::DEVICE);
  ref_graph_updated.Release(util::HOST | util::DEVICE);
}