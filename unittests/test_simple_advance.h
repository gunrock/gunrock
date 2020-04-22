/*
 * @brief Unit tests for dynamic graph operations
 * @file test_simple_advance.h
 */

#include <gunrock/graphio/graphio.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/frontier.cuh>
#include "helper.h"

using namespace gunrock;

// Note: Lambda functions are not in TEST function to avoid this error:
// error: The enclosing parent function ("TestBody") for an extended __host__
// __device__ lambda cannot have private or protected access within its class
template <typename FrontierT, typename SizeT>
void InitFrontierSrc(FrontierT &test_frontier, SizeT nodes) {
  // rng
  std::mt19937 rng(0);
  VertexT advance_src = rng() % nodes;
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

TEST(simpleAdvance, UsingCSRGraph) {
  using FrontierT = app::Frontier<VertexT, SizeT>;

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

  // build a frontier with one source vertex
  FrontierT test_frontier;
  unsigned int num_queues = 2;
  std::vector<app::FrontierType> frontier_type(
      2, app::FrontierType::VERTEX_FRONTIER);
  std::string frontier_name = "test_frontier";
  test_frontier.Init(num_queues, frontier_type.data(), frontier_name,
                     util::DEVICE);

  std::vector<double> queue_factors(num_queues, 2);
  test_frontier.Allocate(ref_csr_graph.nodes, ref_csr_graph.edges,
                         queue_factors);
  test_frontier.Reset(util::DEVICE);

  test_frontier.queue_length = 1;
  InitFrontierSrc(test_frontier, nodes);

  // setup operator parameters
  oprtr::OprtrParameters<WeightedCSRGraphT::CsrT, FrontierT, VertexT>
      oprtr_parameters;
  oprtr_parameters.Init();
  oprtr_parameters.advance_mode = "SIMPLE";

  // call the advance operator
  CallAdvanceOprtr(ref_csr_graph, test_frontier, oprtr_parameters);

  // Get back the resulted frontier length
  test_frontier.work_progress.GetQueueLength(test_frontier.queue_index,
                                             test_frontier.queue_length, false,
                                             oprtr_parameters.stream, true);
}
