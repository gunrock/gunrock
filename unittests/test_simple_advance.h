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
  for (SizeT v = 0; v < output_queue_len; v++)
    result_frontier[v] = output_queue[v];
}
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
}
