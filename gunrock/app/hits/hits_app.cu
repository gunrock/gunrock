// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file hits_app.cu
 *
 * @brief HITS Gunrock Application
 */

#include <gunrock/app/hits/hits_app.cuh>

namespace gunrock {
namespace app {
namespace hits {

template <typename ParametersT>
cudaError_t UseParameters(ParametersT &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));
  GUARD_CU(UseParameters_test(parameters));

  return retval;
}

} // namespace hits
} // namespace app
} // namespace gunrock

/*
 * @brief Simple interface take in a graph in CSR format and return vertex hub and authority scores
 * @param[in]  num_nodes      Number of veritces in the input graph
 * @param[in]  num_edges      Number of edges in the input graph
 * @param[in]  row_offsets    CSR-formatted graph input row offsets
 * @param[in]  col_indices    CSR-formatted graph input column indices
 * @param[in]  max_iter       Maximum number of HITS iterations to perform
 * @param[in]  tol            Algorithm termination tolerance
 * @param[in]  hits_norm      Normalization method
 * @param[out] hub_ranks      Vertex hub scores
 * @param[out] auth ranks     Vertex authority scores
 * @param[in]  allocated_on   Input and output target device, by default CPU
 * \return     double         Return accumulated elapsed times for all iterations
 */
double hits(
    const int                 num_nodes,
    const int                 num_edges,
    const int                 *row_offsets,
    const int                 *col_indices,
    const int                 max_iter,
    const float               tol,
    const int                 hits_norm, 
    float                     *hub_ranks,
    float                     *auth_ranks,
    gunrock::util::Location   allocated_on = gunrock::util::HOST)
{
      return hits<int, int, float>(num_nodes, num_edges, row_offsets, col_indices,
         max_iter, tol, hits_norm, hub_ranks, auth_ranks, allocated_on);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
