// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_test.cu
 *
 * @brief Test related functions for BC
 */

#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <utility>

namespace gunrock {
namespace app {
namespace bc {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the BC result (sigma value and BC value)
 * @tparam SizeT Type of size counters
 * @tparam ValueT Type of values to display
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 * @param[in] quiet Whether to disable print out.
 */
template <typename SizeT, typename ValueT>
void DisplaySolution(ValueT *sigmas, ValueT *bc_values, SizeT nodes,
                     bool quiet = false) {
  if (quiet) return;
  if (nodes > 40) nodes = 40;

  util::PrintMsg("[", true, false);
  for (SizeT v = 0; v < nodes; ++v) {
    util::PrintMsg(std::to_string(v) + ":" + std::to_string(sigmas[v]) + "," +
                       std::to_string(bc_values[v]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference BC implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the BC
 */

template <typename GraphT, typename ValueT = typename GraphT::ValueT,
          typename VertexT = typename GraphT::VertexT>
double CPU_Reference(const GraphT &graph, ValueT *bc_values, ValueT *sigmas,
                     VertexT *source_path, VertexT src, bool quiet = false) {
  typedef typename GraphT::SizeT SizeT;

  for (VertexT v = 0; v < graph.nodes; ++v) {
    bc_values[v] = 0;
    sigmas[v] = v == src ? 1 : 0;
    source_path[v] =
        v == src ? 0 : util::PreDefinedValues<VertexT>::InvalidValue;
  }

  VertexT search_depth = 0;

  std::deque<VertexT> frontier;
  frontier.push_back(src);

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  while (!frontier.empty()) {
    VertexT v = frontier.front();
    frontier.pop_front();
    VertexT neighbor_dist = source_path[v] + 1;

    SizeT edges_begin = graph.row_offsets[v];
    SizeT edges_end = graph.row_offsets[v + 1];

    for (SizeT edge = edges_begin; edge < edges_end; ++edge) {
      VertexT neighbor = graph.column_indices[edge];

      if (!util::isValid(source_path[neighbor])) {
        // if unseen
        source_path[neighbor] = neighbor_dist;
        sigmas[neighbor] += sigmas[v];
        if (search_depth < neighbor_dist) {
          search_depth = neighbor_dist;
        }
        frontier.push_back(neighbor);
      } else {
        // if seen
        if (source_path[neighbor] == source_path[v] + 1) {
          sigmas[neighbor] += sigmas[v];
        }
      }
    }
  }
  search_depth++;

  for (VertexT iter = search_depth - 2; iter > 0; --iter) {
    VertexT cur_level = 0;
    for (VertexT v = 0; v < graph.nodes; ++v) {
      if (source_path[v] == iter) {
        ++cur_level;

        SizeT edges_begin = graph.row_offsets[v];
        SizeT edges_end = graph.row_offsets[v + 1];
        for (SizeT edge = edges_begin; edge < edges_end; ++edge) {
          VertexT neighbor = graph.column_indices[edge];
          if (source_path[neighbor] == iter + 1) {
            bc_values[v] += 1.0f * sigmas[v] / sigmas[neighbor] *
                            (1.0f + bc_values[neighbor]);
          }
        }
      }
    }
  }

  for (VertexT v = 0; v < graph.nodes; ++v) {
    bc_values[v] *= 0.5f;
    //    std::cout << "v=" << v << " | bc_values[v]=" << bc_values[v] <<
    //    std::endl;
  }

  // for (VertexT v =0; v < graph.nodes; ++v) {
  //    std::cout << "v=" << v << " | sigmas[v]=" << sigmas[v] << std::endl;
  //}

  // for (VertexT v =0; v < graph.nodes; ++v) {
  //    std::cout << "v=" << v << " | source_path[v]=" << source_path[v] <<
  //    std::endl;
  //}

  cpu_timer.Stop();
  return cpu_timer.ElapsedMillis();
}

/**
 * @brief Validation of BC results
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
    util::Parameters &parameters, GraphT &graph, typename GraphT::VertexT src,
    ValueT *h_bc_values, ValueT *h_sigmas, typename GraphT::VertexT *h_labels,
    ValueT *ref_bc_values = NULL, ValueT *ref_sigmas = NULL,
    typename GraphT::VertexT *ref_labels = NULL, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  SizeT num_vertices = graph.nodes;
  bool quiet = parameters.Get<bool>("quiet");

  if (ref_bc_values != NULL) {
    util::PrintMsg("BC value validity:", !quiet, false);
    SizeT errors_num = util::CompareResults(h_bc_values, ref_bc_values,
                                            num_vertices, true, quiet);
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;
    }
  }

  if (ref_sigmas != NULL) {
    util::PrintMsg("Sigma validity:", !quiet, false);
    SizeT errors_num =
        util::CompareResults(h_sigmas, ref_sigmas, num_vertices, true, quiet);
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;
    }
  }

  if (ref_labels != NULL) {
    util::PrintMsg("Label validity:", !quiet, false);
    SizeT errors_num =
        util::CompareResults(h_labels, ref_labels, num_vertices, true, quiet);
    if (errors_num > 0) {
      util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
      num_errors += errors_num;
    }
  }

  if (!quiet && verbose) {
    // Display Solution
    DisplaySolution(h_bc_values, h_sigmas, num_vertices, quiet);
  }

  return num_errors;
}

}  // namespace bc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
