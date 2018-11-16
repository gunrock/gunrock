// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @fille
 * tc_test.cuh
 *
 * @brief Test related functions for TC
 */

#pragma once

#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <vector>
#include <utility>

namespace gunrock {
namespace app {
namespace tc {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

 cudaError_t UseParameters_test(util::Parameters &parameters)
 {
     cudaError_t retval = cudaSuccess;

     GUARD_CU(parameters.Use<uint32_t>(
         "omp-threads",
         util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
         0,
         "Number of threads for parallel omp louvain implementation; 0 for default.",
         __FILE__, __LINE__));

     GUARD_CU(parameters.Use<int>(
         "omp-runs",
         util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
         1,
         "Number of runs for parallel omp louvain implementation.",
         __FILE__, __LINE__));

     return retval;
 }

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length)
{
    if (length > 40)
        length = 40;

    util::PrintMsg("[", true, false);
    for (SizeT i = 0; i < length; ++i)
    {
        util::PrintMsg(std::to_string(i) + ":"
            + std::to_string(array[i]) + " ", true, false);
    }
    util::PrintMsg("]");
}

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SSSP ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the SSSP
 */
template <
    typename GraphT,
    typename VertexT = typename GraphT::VertexT>
double CPU_Reference(
  util::Parameters         &parameters,
           GraphT          &graph,
           VertexT         *tc_count)
{
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    return elapsed;
}
/**
 * @brief Validation of TC results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_distances   Computed distances from the source to each vertex
 * @param[in]  h_preds       Computed predecessors for each vertex
 * @param[in]  ref_distances Reference distances from the source to each vertex
 * @param[in]  ref_preds     Reference predecessors for each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */

template <
    typename GraphT,
    typename VertexT = typename GraphT::VertexT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
                    VertexT   *h_tc_count,
                    VertexT   *ref_tc,
         bool      verbose       = true)
{
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    return num_errors;
}

} // namespace tc
} // namespace app
} // namespace gunrock
