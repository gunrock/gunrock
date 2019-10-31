// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file ss_app.cu
 *
 * @brief subgraph matching (SM) application
 */

#include <gunrock/app/sm/sm_app.cuh>

namespace gunrock {
namespace app {
namespace sm {

template <typename ParametersT>
cudaError_t UseParameters(ParametersT &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  GUARD_CU(UseParameters_test(parameters));

  GUARD_CU(parameters.Use<unsigned int>(
              "num-subgraphs",
              util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::INTERNAL_PARAMETER,
              0, "number of matched subgraphs", __FILE__, __LINE__));

  return retval;
}

} // namespace sm
} // namespace app
} // namespace gunrock

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform SM
 * @param[out] subgraphs   Return number of subgraphs
 * \return     double      Return accumulated elapsed times for all runs
 */
double sm(
    const int            num_nodes,
    const int            num_edges,
    const int           *row_offsets,
    const int           *col_indices,
    const unsigned long *edge_values,
    const int            num_runs,
          int           *subgraphs)
{
    return sm(num_nodes, num_edges, row_offsets, col_indices,
        edge_values, 1 /* num_runs */, subgraphs);
}

/*
 * @brief Simple interface take in graph as Gunrock format
 * @param[in]  query_graph Query graph to be searched
 * @param[in]  data_graph  data graph to be searched on
 * @param[in]  num_runs    Number of runs to perform SM
 * @param[out] subgraphs   Return number of subgraphs
 * \return     double      Return accumulated elapsed times for all runs
 */
double nv_sm(
    gunrock::app::TestGraph<int, int, unsigned long,
    gunrock::graph::HAS_CSR> &query_graph,
    gunrock::app::TestGraph<int, int, unsigned long,
    gunrock::graph::HAS_CSR> &data_graph,
    const int            num_runs,
          int           *subgraphs)
{
    return nv_sm(query_graph, data_graph, 1 /* num_runs */, subgraphs);
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
