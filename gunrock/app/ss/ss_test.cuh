// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * ss_test.cu
 *
 * @brief Test related functions for SS
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
namespace ss {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

cudaError_t UseParameters_test(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<uint32_t>(
      "omp-threads",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "Number of threads for parallel omp scan statistics implementation; 0 "
      "for default.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "omp-runs",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1, "Number of runs for parallel omp scan statistics implementation.",
      __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Displays the SS result (i.e., statistic values for each node)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length) {
  if (length > 40) length = 40;

  util::PrintMsg("[", true, false);
  for (SizeT i = 0; i < length; ++i) {
    util::PrintMsg(std::to_string(i) + ":" + std::to_string(array[i]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/******************************************************************************
 * SS Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SS implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the SS
 */
template <typename GraphT, typename VertexT = typename GraphT::VertexT>
double CPU_Reference(util::Parameters &parameters, GraphT &graph,
                     VertexT *scan_stats) {
  printf("CPU_Reference: start\n");

  // In pseudocode:
  //
  // Init scan_stats[i] = degree(i)
  // For (u, v) in graph:
  //   u_neibs = get_neibs(u)
  //   v_neibs = get_neibs(v)
  //   For n in intersect(u_neibs, v_neibs):
  //     scan_stats[n] += 1

  typedef typename GraphT::SizeT SizeT;

  util::CpuTimer cpu_timer;
  float total_time = 0.0;
  int num_iter = parameters.Get<int>("num-runs");
  // Run 10 iterations
  for (int iter = 0; iter < num_iter; ++iter) {
    cpu_timer.Start();
    // Initialize scan stat as degree of nodes
    for (VertexT i = 0; i < graph.nodes; i++) {
      scan_stats[i] = graph.GetNeighborListLength(i);
    }

    // For each node
    for (VertexT src = 0; src < graph.nodes; src++) {
      SizeT src_num_neighbors = graph.GetNeighborListLength(src);
      if (src_num_neighbors > 0) {
        SizeT src_edge_start = graph.GetNeighborListOffset(src);
        SizeT src_edge_end = src_edge_start + src_num_neighbors;

        // Iterate over outgoing edges
        for (SizeT src_edge_idx = src_edge_start; src_edge_idx < src_edge_end;
             src_edge_idx++) {
          VertexT dst = graph.GetEdgeDest(src_edge_idx);
          if (src < dst) {  // Avoid double counting.  This also implies we only
                            // support undirected graphs.

            SizeT dst_num_neighbors = graph.GetNeighborListLength(dst);
            if (dst_num_neighbors > 0) {
              SizeT dst_edge_start = graph.GetNeighborListOffset(dst);
              SizeT dst_edge_end = dst_edge_start + dst_num_neighbors;

              // Find nodes that are neighbors of both `src` and `dst`
              // Note: This assumes that neighbor lists are sorted
              int src_offset = src_edge_start;
              int dst_offset = dst_edge_start;
              while (dst_offset < dst_edge_end && src_offset < src_edge_end) {
                VertexT dst_neib = graph.GetEdgeDest(dst_offset);
                VertexT src_neib = graph.GetEdgeDest(src_offset);
                if (dst_neib == src_neib) {
                  scan_stats[src_neib]++;
                  dst_offset++;
                  src_offset++;
                } else if (dst_neib < src_neib) {
                  dst_offset++;
                } else if (src_neib < dst_neib) {
                  src_offset++;
                }
              }
            }
          }
        }
      }
    }
    cpu_timer.Stop();
    total_time += cpu_timer.ElapsedMillis();
  }
  float elapsed = total_time / num_iter;

  printf("CPU_Reference: done\n");

  return elapsed;
}

/**
 * @brief Validation of SS results
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
template <typename GraphT, typename VertexT = typename GraphT::VertexT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph, VertexT *h_scan_stat,
                                        VertexT *ref_scan_stat,
                                        bool verbose = true) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  std::cerr << "Validate_Results" << std::endl;
  bool quiet = parameters.Get<bool>("quiet");
  if (!quiet && verbose) {
    for (int i = 0; i < graph.nodes; i++) {
      std::cerr << i << " " << ref_scan_stat[i] << " " << h_scan_stat[i]
                << std::endl;
    }
  }

  SizeT num_errors = 0;

  // Verify the result
  util::PrintMsg("Scan statistics Validity: ", !quiet, false);
  num_errors = util::CompareResults(h_scan_stat, ref_scan_stat, graph.nodes,
                                    true, quiet);

  if (num_errors > 0) {
    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    return num_errors;
  } else {
    util::PrintMsg("PASS", !quiet);
  }

  if (!quiet && verbose) {
    util::PrintMsg("scan_statistics: ");
    DisplaySolution(h_scan_stat, graph.nodes);
  }

  return num_errors;
}

}  // namespace ss
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
