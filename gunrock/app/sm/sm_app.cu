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

cudaError_t UseParameters(util::Parameters &parameters) {
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
 * @param[in]  num_nodes         Number of veritces in the input data graph
 * @param[in]  num_edges         Number of edges in the input data graph
 * @param[in]  row_offsets       CSR-formatted data graph input row offsets
 * @param[in]  col_indices       CSR-formatted data graph input column indices
 * @param[in]  num_query_nodes   Number of veritces in the input query graph
 * @param[in]  num_query_edges   Number of edges in the input query graph
 * @param[in]  query_row_offsets CSR-formatted graph input query row offsets
 * @param[in]  query_col_indices CSR-formatted graph input query column indices
 * @param[in]  num_runs          Number of runs to perform SM
 * @param[out] subgraphs         Return number of subgraphs
 * @param[out] list_subgraphs    Return list of subgraphs
 * @param[in]  allocated_on      Input and output target device, by default CPU
 * \return     double            Return accumulated elapsed times for all runs
 */
double sm(
    const int                 num_nodes,
    const int                 num_edges,
    const int                *row_offsets,
    const int                *col_indices,
    const int                 num_query_nodes,
    const int                 num_query_edges,
    const int                *query_row_offsets,
    const int                *query_col_indices,
    const int                 num_runs,
    unsigned long            *subgraphs,
    unsigned long            **list_subgraphs,
    gunrock::util::Location   allocated_on = gunrock::util::HOST)
{
    return sm<int, int>(num_nodes, num_edges, row_offsets, col_indices,
        num_query_nodes, num_query_edges, query_row_offsets,
        query_col_indices, num_runs, subgraphs, list_subgraphs, allocated_on);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
