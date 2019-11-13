// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kcore_test.cu
 *
 * @brief Test related functions for K-Core
 */

#pragma once

#include <map>
#include <unordered_map>
#include <set>

namespace gunrock {
namespace app {
namespace kcore {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
/**
 * @brief Displays the k-core result (i.e. number of cores of vertices)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] num_cores number of cores for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <typename T, typename SizeT>
void DisplaySolution(T *num_cores, SizeT num_nodes) {
  if (num_nodes > 40) num_nodes = 40;

  util::PrintMsg("[", true, false);
  for (SizeT i = 0; i < num_nodes; ++i) {
    util::PrintMsg(std::to_string(i) + ":" + std::to_string(num_cores[i]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/******************************************************************************
 * K-Core Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference K-Core implementation
 * @tparam      GraphT        Type of the graph
 * @param[in]   parameters    Input parameters
 * @param[in]   graph         Input graph
 * @param[out]  num_cores     Number of cores for each vertex
 * \return      double        Time taken for the K-Core implementation
 */
template <typename GraphT>
double CPU_Reference(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::SizeT *num_cores) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  SizeT nodes = graph.nodes;
  SizeT edges = graph.edges;
  SizeT *out_degrees = (SizeT *)malloc(sizeof(SizeT) * nodes);
  bool *to_remove = (bool*)malloc(sizeof(bool) * nodes);
  util::CpuTimer cpu_timer;

  #pragma omp parallel for
    for (VertexT v = 0; v < nodes; ++v) {
      num_cores[v] = 0;
    }
  
  #pragma omp parallel for
    for (SizeT e = 0; e < edges; ++e) {
      VertexT v, u;
      graph.CsrT::GetEdgeSrcDest(e, v, u);
  #pragma omp atomic
      out_degrees[v] += 1;
    }
  
  cpu_timer.Start();
  num_cores[nodes] = -1; // largest core is the last element
  SizeT num_to_remove = 0;
  SizeT num_to_remain = nodes;

  for (SizeT k = 1; k <= nodes; ++k) {
    #pragma omp parallel for
    for (VertexT v = 0; v < nodes; ++v) {
      to_remove[v] = false;
    }
    
    while (true) {
      num_to_remove = 0;
      num_to_remain = 0;
      #pragma omp parallel for
        for (VertexT v = 0; v < nodes; ++v) {
          if (!to_remove[v] && out_degrees[v] < k) {
            num_cores[v] = k - 1;
            out_degrees[v] = 0;
            to_remove[v] = true;
            num_to_remove++;
          }
          if (out_degrees[v] >= k) {
            num_to_remain++;
          }
        }
      if (num_to_remove == 0) break;
      else {
        #pragma omp parallel for
          for (SizeT e = 0; e < edges; ++e) {
            VertexT src, dest;
            graph.CsrT::GetEdgeSrcDest(e, src, dest);
            if (to_remove[src]) {
              #pragma omp atomic
                out_degrees[dest] -= 1;
            }
          }
      }
    }
    if (num_to_remain == 0) {
      num_cores[nodes] = k - 1;
      break;
    }
  }
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  free(out_degrees);
  out_degrees = NULL;
  free(to_remove);
  to_remove = NULL;

  return elapsed;
}

/**
 * @brief Validation of K-Core results
 * @tparam     GraphT        Type of the graph
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  h_num_cores   Computed k-core numbers
 * @param[in]  ref_num_cores Reference k-core numbers
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph,
    typename GraphT::VertexT *h_num_cores,
    typename GraphT::VertexT *ref_num_cores = NULL, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  
  bool quiet = parameters.Get<bool>("quiet");
  SizeT nodes = graph.nodes;
  SizeT num_errors = 0;

  // Verify the results
  if (ref_num_cores) {
    util::PrintMsg("K-Core Number Validity: ", !quiet, false);
    SizeT errors_num = 
      util::CompareResults(h_num_cores, ref_num_cores, nodes, /*verbose=*/true, quiet);
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;

      VertexT min_mismatch_label = util::PreDefinedValues<VertexT>::MaxValue;
      VertexT min_mismatch_vertex =
          util::PreDefinedValues<VertexT>::InvalidValue;
      for (VertexT v = 0; v < nodes; v++) {
        if (h_num_cores[v] == ref_num_cores[v]) continue;
        if (h_num_cores[v] >= min_mismatch_label) continue;
        min_mismatch_label = h_num_cores[v];
        min_mismatch_vertex = v;
      }
      util::PrintMsg(
          "First mismatch: ref_num_cores[" + std::to_string(min_mismatch_vertex) +
              "] (" + std::to_string(ref_num_cores[min_mismatch_vertex]) +
              ") != h_num_cores[" + std::to_string(min_mismatch_vertex) + "] (" +
              std::to_string(h_num_cores[min_mismatch_vertex]) + ")",
          !quiet);
    }
  }

  if (!quiet && verbose) {
    // Display Solution
    util::PrintMsg("First few num_cores of the GPU result:");
    DisplaySolution(h_num_cores, nodes);
    if (ref_num_cores != NULL) {
      util::PrintMsg("First few num_cores of the reference CPU result.");
      DisplaySolution(ref_num_cores, nodes);
    }
    util::PrintMsg("Largest K is " + std::to_string(h_num_cores[nodes]));
  }
}

}  // namespace kcore
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
