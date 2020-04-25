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

void RandomWeightedGraphToCsr(DynamicHostGraphT &ref_dyn,
                              WeightedCSRGraphT &ref_graph,
                              int const_weights = 0) {
  // rng
  std::mt19937 rng(0);

  auto ref_csr_graph = ref_graph.csr();

  // Populate the CSR with the adj. list
  SizeT cur_offset = 0;
  for (auto v = 0; v < ref_dyn.size(); v++) {
    auto &v_edges = ref_dyn[v];
    ref_csr_graph.row_offsets[v] = cur_offset;
    for (const auto &e : v_edges) {
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
void GenerateRandomWeightedGraph(DynamicHostGraphT &ref_dyn, SizeT nodes,
                                 SizeT edges, SizeT edges_per_node,
                                 bool undirected_graph) {
  // rng
  std::mt19937 rng(0);

  // generate a random reference  graph
  for (VertexT v = 0; v < nodes; v++) {
    auto &v_edges = ref_dyn[v];
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
void CompareWeightedCSRs(WeightedCSRGraphT &ref_graph,
                         WeightedDYNGraphT &result_graph) {
  // sort both CSR graphs
  auto &result_csr_graph = result_graph.csr();
  auto &ref_csr_graph = ref_graph.csr();

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

// Note: Lambda functions are not in TEST function to avoid this error:
// error: The enclosing parent function ("TestBody") for an extended __host__
// __device__ lambda cannot have private or protected access within its class
template <typename FrontierT, typename VertexT>
void InitFrontierSrc(FrontierT &test_frontier, VertexT &advance_src) {
  test_frontier.V_Q()->ForEach(
      [advance_src] __host__ __device__(VertexT & v) { v = advance_src; }, 1,
      util::DEVICE, 0);
}

template <typename GraphT, typename FrontierT, typename ParameterT>
void CallAdvanceOprtr(GraphT &ref_csr_graph, FrontierT &test_frontier,
                      ParameterT &oprtr_parameters) {
  // the advance operation
  auto advance_op = [] __host__ __device__(
                        const VertexT &src, VertexT &dest,
                        const ValueT &edge_value, const VertexT &input_item,
                        const SizeT &input_pos,
                        SizeT &output_pos) -> bool { return true; };
  oprtr::Advance<oprtr::OprtrType_V2V>(ref_csr_graph, test_frontier.V_Q(),
                                       test_frontier.Next_V_Q(),
                                       oprtr_parameters, advance_op);
}
template <typename GraphT, typename VertexT>
void advanceTester(std::string advance_mode, GraphT &graph, VertexT advance_src,
                   std::vector<VertexT> &result_frontier) {
  using FrontierT = app::Frontier<VertexT, SizeT>;

  // build a frontier with one source vertex
  FrontierT test_frontier;
  unsigned int num_queues = 2;
  std::vector<app::FrontierType> frontier_type(
      2, app::FrontierType::VERTEX_FRONTIER);
  std::string frontier_name = "test_frontier";
  test_frontier.Init(num_queues, frontier_type.data(), frontier_name,
                     util::DEVICE);

  std::vector<double> queue_factors(num_queues, 2);
  test_frontier.Allocate(graph.nodes, graph.edges, queue_factors);
  test_frontier.Reset(util::DEVICE);

  // generate initial frontier
  SizeT num_srcs = 1;
  test_frontier.queue_length = num_srcs;
  InitFrontierSrc(test_frontier, advance_src);

  // setup operator parameters
  oprtr::OprtrParameters<GraphT, FrontierT, VertexT> oprtr_parameters;
  oprtr_parameters.Init();
  oprtr_parameters.advance_mode = advance_mode;
  oprtr_parameters.frontier = &test_frontier;

  // call the advance operator
  CallAdvanceOprtr(graph, test_frontier, oprtr_parameters);

  // Get back the resulted frontier length
  test_frontier.work_progress.GetQueueLength(test_frontier.queue_index,
                                             test_frontier.queue_length, false,
                                             oprtr_parameters.stream, true);
  auto output_queue = *test_frontier.V_Q();
  auto output_queue_len = test_frontier.queue_length;
  output_queue.Move(util::DEVICE, util::HOST);

  // store results
  result_frontier.resize(output_queue_len);
  for (SizeT v = 0; v < output_queue_len; v++) {
    result_frontier[v] = output_queue[v];
  }

  test_frontier.Release();
}