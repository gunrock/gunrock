// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * lp_test.cu
 *
 * @brief Test related functions for BFS
 */

#pragma once

#include <vector>

namespace gunrock {
namespace app {
namespace lp {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the BFS result (i.e., hop distance from source)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] values values to display.
 * @param[in] num_nodes Number of values to display.
 */
template <typename T, typename SizeT>
void DisplaySolution(T *values, SizeT length) {
  if (length > 40) length = 40;

  util::PrintMsg("[", true, false);
  for (SizeT i = 0; i < length; ++i) {
    util::PrintMsg(std::to_string(i) + ":" + std::to_string(values[i]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/**
 * @brief Simple CPU-based reference BFS implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      LabelT        Type of the labels
 * @param[in]   graph         Input graph
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[out]  labels        Computed hop distances from the source to each
 * vertex
 * \return      double        Time taken for the BFS
 */
template <typename GraphT, typename LabelT = typename GraphT::VertexT>
double CPU_Reference(const GraphT &graph, typename GraphT::VertexT src,
                     bool quiet, LabelT *labels,
                     typename GraphT::VertexT &num_iters) {
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;


  // Initialize labels
  for (VertexT v = 0; v < graph.nodes; v++) {
    labels[v] = util::PreDefinedValues<LabelT>::MaxValue;
  }
  labels[src] = 0;

  // ping-pong frontiers
  VertexT iter = 0;
  std::vector<VertexT> frontiers[2];
  frontiers[0].push_back(src);

  // Perform BFS
  util::CpuTimer cpu_timer;
  cpu_timer.Start();
  while (!frontiers[iter % 2].empty()) {
    auto &curr_frontier = frontiers[iter % 2];
    auto &next_frontier = frontiers[(iter + 1) % 2];
    next_frontier.clear();
    iter++;

    for (auto v : curr_frontier) {
      // Locate adjacency list
      SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
      SizeT e_end = e_start + graph.CsrT::GetNeighborListLength(v);
      for (SizeT e = e_start; e < e_end; e++) {
        VertexT u = graph.CsrT::GetEdgeDest(e);
        if (iter < labels[u]) {
          labels[u] = iter;
          next_frontier.push_back(u);
        }
      }
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  // util::PrintMsg("CPU BFS finished in " + std::to_string(elapsed)
  //    + " msec. cpu_search_depth: " + std::to_string(iter), !quiet);
  frontiers[0].clear();
  frontiers[1].clear();
  num_iters = iter;
  return elapsed;
}

/**
 * @brief Validation of BFS results
 * @tparam     GraphT        Type of the graph
 * @tparam     LabelT        Type of the labels
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_labels      Computed hop distances from the source to each
 * vertex
 * @param[in]  ref_labels    Reference labels from the source to each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename LabelT = typename GraphT::LabelT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph, typename GraphT::VertexT src,
    LabelT *h_labels, LabelT *ref_labels = NULL,
    bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  int test = parameters.Get<int>("test");

  if (test == 1) {
    // run test for test_lp_1.mtx
    ref_labels = (LabelT*)malloc(4*sizeof(LabelT));
    for (int i = 0; i < 4; i++) {
      ref_labels[i] = 0;
    }   
  }
  else if (test == 2) {
    // TODO
    // create another test for disconnected components
  }

  SizeT num_errors = 0;
  // bool quick = parameters.Get<bool>("quick");
  bool quiet = parameters.Get<bool>("quiet");

  // Verify the labels
  if (ref_labels != NULL) {
    util::PrintMsg("Label Validity: ", !quiet, false);
    SizeT errors_num =
        util::CompareResults(h_labels, ref_labels, graph.nodes, true, quiet);
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;

      LabelT min_mismatch_label = util::PreDefinedValues<LabelT>::MaxValue;
      VertexT min_mismatch_vertex =
          util::PreDefinedValues<VertexT>::InvalidValue;
      for (VertexT v = 0; v < graph.nodes; v++) {
        if (h_labels[v] == ref_labels[v]) continue;
        if (h_labels[v] >= min_mismatch_label) continue;
        min_mismatch_label = h_labels[v];
        min_mismatch_vertex = v;
      }
      util::PrintMsg(
          "First mismatch: ref_labels[" + std::to_string(min_mismatch_vertex) +
              "] (" + std::to_string(ref_labels[min_mismatch_vertex]) +
              ") != h_labels[" + std::to_string(min_mismatch_vertex) + "] (" +
              std::to_string(h_labels[min_mismatch_vertex]) + ")",
          !quiet);
    }
  } else if (ref_labels == NULL) {
    util::PrintMsg("Label Validity: ", !quiet, false);
    SizeT errors_num = 0;
    for (VertexT v = 0; v < graph.nodes; v++) {
      LabelT v_label = h_labels[v];
      if (!util::isValid(v_label)) continue;
      SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
      SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
      SizeT e_end = e_start + num_neighbors;
      for (SizeT e = e_start; e < e_end; e++) {
        VertexT u = graph.CsrT::GetEdgeDest(e);
        LabelT u_label = h_labels[u];
        if (v_label + 1 >= u_label) continue;
        errors_num++;
        if (errors_num > 1) continue;

        util::PrintMsg(
            "FAIL: v[" + std::to_string(v) + "] (" + std::to_string(v_label) +
                ") + e[" + std::to_string(e) + "] (1) < u[" +
                std::to_string(u) + "] (" + std::to_string(u_label) + ")",
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
    util::PrintMsg("First 40 labels of the GPU result:");
    DisplaySolution(h_labels, graph.nodes);
    if (ref_labels != NULL) {
      util::PrintMsg("First 40 distances of the reference CPU result.");
      DisplaySolution(ref_labels, graph.nodes);
    }
    util::PrintMsg("");
  }

  

  return num_errors;
}

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
