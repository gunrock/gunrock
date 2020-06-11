
// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * validation.cuh
 *
 * @brief Compare dynmic graph and sorted CSR for validation.
 */

#include <gunrock/util/error_utils.cuh>

template <typename CsrT, typename DynT>
int CompareWeightedDynCSR(CsrT& graph_csr, DynT& graph_dyn, bool quite) {
  using SizeT = typename CsrT::SizeT;
  using ValueT = typename CsrT::ValueT;
  using VertexT = typename CsrT::VertexT;
  using WeightedCsrT = gunrock::app::TestGraph<VertexT, SizeT, ValueT,
                                               gunrock::graph::HAS_EDGE_VALUES |
                                                   gunrock::graph::HAS_CSR>;

  SizeT nodes = graph_csr.nodes;
  SizeT edges = graph_csr.edges;

  WeightedCsrT res_graph_dyn;
  auto& res_graph_csr = res_graph_dyn.csr();
  res_graph_csr.Allocate(nodes, edges,
                         gunrock::util::HOST | gunrock::util::DEVICE);
  graph_dyn.ToCsr(res_graph_csr);
  res_graph_csr.Move(gunrock::util::DEVICE, gunrock::util::HOST);
  res_graph_csr.Sort();

  // Make sure number of edges is the same:
  if (graph_csr.row_offsets[nodes] != res_graph_csr.row_offsets[nodes]) {
    gunrock::util::PrintMsg("Number of edges mismatch.", !quite);
    res_graph_dyn.Release();
    return 1;
  }

  // Now validate each adjacency list
  for (auto v = 0; v < nodes; v++) {
    if (graph_csr.row_offsets[v] != res_graph_csr.row_offsets[v]) {
      gunrock::util::PrintMsg("Number of edges for vertex mismatch.", !quite);
      res_graph_dyn.Release();
      return 1;
    }

    auto start_eid = graph_csr.row_offsets[v];
    auto end_eid = graph_csr.row_offsets[v + 1];

    for (auto eid = start_eid; eid < end_eid; eid++) {
      if (graph_csr.column_indices[eid] != res_graph_csr.column_indices[eid]) {
        gunrock::util::PrintMsg("Edge id mismatch.", !quite);
        res_graph_dyn.Release();
        return 1;
      }
      if (graph_csr.edge_values[eid] != res_graph_csr.edge_values[eid]) {
        gunrock::util::PrintMsg("Edge value mismatch.", !quite);
        res_graph_dyn.Release();
        return 1;
      }
    }
  }
  res_graph_dyn.Release();
  return 0;
}