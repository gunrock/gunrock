// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * vn_test.cu
 *
 * @brief Test related functions for vn
 */

#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <utility>

namespace gunrock {
namespace app {
namespace vn {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the vn result (i.e., distance from source)
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
 * vn Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference vn implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   srcs          The source vertex array
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the vn
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CPU_Reference(const GraphT &graph, ValueT *distances,
                     typename GraphT::VertexT *preds,
                     typename GraphT::VertexT *srcs,
                     typename GraphT::SizeT num_srcs, bool quiet,
                     bool mark_preds) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef std::pair<VertexT, ValueT> PairT;
  struct GreaterT {
    bool operator()(const PairT &lhs, const PairT &rhs) {
      return lhs.second > rhs.second;
    }
  };
  typedef std::priority_queue<PairT, std::vector<PairT>, GreaterT> PqT;

  for (VertexT v = 0; v < graph.nodes; v++) {
    distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
    if (mark_preds && preds != NULL)
      preds[v] = util::PreDefinedValues<VertexT>::InvalidValue;
  }

  PqT pq;

  for (int i = 0; i < num_srcs; ++i) {
    distances[srcs[i]] = 0;
    if (mark_preds && preds != NULL) preds[srcs[i]] = srcs[i];

    pq.push(std::make_pair(srcs[i], 0));
  }

  util::CpuTimer cpu_timer;
  cpu_timer.Start();
  while (!pq.empty()) {
    auto pair = pq.top();
    pq.pop();
    VertexT v = pair.first;
    ValueT v_distance = pair.second;
    if (v_distance > distances[v]) continue;

    SizeT e_start = graph.GetNeighborListOffset(v);
    SizeT e_end = e_start + graph.GetNeighborListLength(v);
    for (SizeT e = e_start; e < e_end; e++) {
      VertexT u = graph.GetEdgeDest(e);
      ValueT u_distance = v_distance + graph.edge_values[e];
      if (u_distance < distances[u]) {
        distances[u] = u_distance;
        if (mark_preds && preds != NULL) preds[u] = v;
        pq.push(std::make_pair(u, u_distance));
      }
    }
  }
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  // util::PrintMsg("CPU VN finished in " + std::to_string(elapsed)
  //    + " msec.", !quiet);

  return elapsed;
}

/**
 * @brief Validation of vn results
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
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph, typename GraphT::VertexT *srcs,
    ValueT *h_distances, typename GraphT::VertexT *h_preds,
    ValueT *ref_distances = NULL, typename GraphT::VertexT *ref_preds = NULL,
    bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  SizeT num_errors = 0;
  // bool quick = parameters.Get<bool>("quick");
  bool quiet = parameters.Get<bool>("quiet");
  bool mark_pred = parameters.Get<bool>("mark-pred");
  SizeT num_srcs = sizeof(srcs) / sizeof(srcs[0]);

  // Verify the result
  if (ref_distances != NULL) {
    for (VertexT v = 0; v < graph.nodes; v++) {
      if (!util::isValid(ref_distances[v]))
        ref_distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
    }

    util::PrintMsg("Distance Validity: ", !quiet, false);
    SizeT errors_num = util::CompareResults(h_distances, ref_distances,
                                            graph.nodes, true, quiet);
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;
    }
  } else if (ref_distances == NULL) {
    util::PrintMsg("Distance Validity: ", !quiet, false);
    SizeT errors_num = 0;
    for (VertexT v = 0; v < graph.nodes; v++) {
      ValueT v_distance = h_distances[v];
      if (!util::isValid(v_distance)) continue;
      SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
      SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
      SizeT e_end = e_start + num_neighbors;
      for (SizeT e = e_start; e < e_end; e++) {
        VertexT u = graph.CsrT::GetEdgeDest(e);
        ValueT u_distance = h_distances[u];
        ValueT e_value = graph.CsrT::edge_values[e];
        if (v_distance + e_value >= u_distance) continue;
        errors_num++;
        if (errors_num > 1) continue;

        util::PrintMsg("FAIL: v[" + std::to_string(v) + "] (" +
                           std::to_string(v_distance) + ") + e[" +
                           std::to_string(e) + "] (" + std::to_string(e_value) +
                           ") < u[" + std::to_string(u) + "] (" +
                           std::to_string(u_distance) + ")",
                       !quiet);
      }
    }
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;
    } else {
      util::PrintMsg("PASS", !quiet);
    }
  }

  if (!quiet && verbose) {
    // Display Solution
    util::PrintMsg("First 40 distances of the GPU result:");
    DisplaySolution(h_distances, graph.nodes);
    if (ref_distances != NULL) {
      util::PrintMsg("First 40 distances of the reference CPU result.");
      DisplaySolution(ref_distances, graph.nodes);
    }
    util::PrintMsg("");
  }

  if (mark_pred) {
    util::PrintMsg("Predecessors Validity: ", !quiet, false);
    SizeT errors_num = 0;
    for (VertexT v = 0; v < graph.nodes; v++) {
      VertexT pred = h_preds[v];

      bool do_continue;
      do_continue = false;
      for (SizeT i = 0; i < num_srcs; ++i) {
        if (v == srcs[i]) {
          do_continue = true;
        }
      }

      if (!util::isValid(pred) || do_continue) continue;

      ValueT v_distance = h_distances[v];
      if (v_distance == util::PreDefinedValues<ValueT>::MaxValue) continue;
      ValueT pred_distance = h_distances[pred];
      bool edge_found = false;
      SizeT edge_start = graph.CsrT::GetNeighborListOffset(pred);
      SizeT num_neighbors = graph.CsrT::GetNeighborListLength(pred);

      for (SizeT e = edge_start; e < edge_start + num_neighbors; e++) {
        if (v == graph.CsrT::GetEdgeDest(e) &&
            std::abs((pred_distance + graph.CsrT::edge_values[e] - v_distance) *
                     1.0) < 1e-6) {
          edge_found = true;
          break;
        }
      }
      if (edge_found) continue;
      errors_num++;
      if (errors_num > 1) continue;

      util::PrintMsg("FAIL: [" + std::to_string(pred) + "] (" +
                         std::to_string(pred_distance) + ") -> [" +
                         std::to_string(v) + "] (" +
                         std::to_string(v_distance) +
                         ") can't find the corresponding edge.",
                     !quiet);
    }
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;
    } else {
      util::PrintMsg("PASS", !quiet);
    }
  }

  if (!quiet && mark_pred && verbose) {
    util::PrintMsg("First 40 preds of the GPU result:");
    DisplaySolution(h_preds, graph.nodes);
    if (ref_preds != NULL) {
      util::PrintMsg(
          "First 40 preds of the reference CPU result "
          "(could be different because the paths are not unique):");
      DisplaySolution(ref_preds, graph.nodes);
    }
    util::PrintMsg("");
  }

  return num_errors;
}

}  // namespace vn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
