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
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[out] hub_ranks   Vertex hub scores
 * @param[out] auth ranks  Vertex authority scores
 * \return     double      Return accumulated elapsed times for all iterations
 */
double hits(
    const int        num_nodes,
    const int        num_edges,
    const int       *row_offsets,
    const int     *col_indices,
    const int          num_iter,
    float            *hub_ranks,
    float            *auth_ranks)
{
      return hits_template(num_nodes, num_edges, row_offsets, col_indices,
         num_iter, hub_ranks, auth_ranks);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
