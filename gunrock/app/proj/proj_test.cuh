// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * proj_test.cu
 *
 * @brief Test related functions for proj
 */

#pragma once

namespace gunrock {
namespace app {
namespace proj {

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference proj ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(const GraphT &graph, typename GraphT::ValueT *projections,
                     bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  for (SizeT edge_idx = 0; edge_idx < graph.nodes * graph.nodes; edge_idx++) {
    projections[edge_idx] = 0;
  }

  for (SizeT node = 0; node < graph.nodes; node++) {
    SizeT num_neighbors = graph.GetNeighborListLength(node);
    SizeT node_offset = graph.GetNeighborListOffset(node);

    for (SizeT offset_1 = 0; offset_1 < num_neighbors; offset_1++) {
      VertexT neib1 = graph.GetEdgeDest(node_offset + offset_1);

      for (SizeT offset_2 = 0; offset_2 < num_neighbors; offset_2++) {
        VertexT neib2 = graph.GetEdgeDest(node_offset + offset_2);

        if (neib1 != neib2) {
          ValueT edge_weight =
              (ValueT)1.0;  // Could so more complex functions of edge weights
          SizeT edge_idx = (SizeT)neib1 * graph.nodes + (SizeT)neib2;
          projections[edge_idx] += edge_weight;
        }
      }
    }
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  return elapsed;
}

/**
 * @brief Validation of proj results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph,
    typename GraphT::ValueT *h_projections,
    typename GraphT::ValueT *ref_projections = NULL, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");

  SizeT edge_counter = 0;
  for (SizeT v = 0; v < graph.nodes * graph.nodes; ++v) {
    if (h_projections[v] != 0) {
      edge_counter++;
    }
  }
  printf("edge_counter=%d\n", edge_counter);

  if (ref_projections != NULL) {
    for (SizeT v = 0; v < graph.nodes * graph.nodes; ++v) {
      if (ref_projections[v] != 0) {
        int row = (int)(v / graph.nodes);
        int col = v % graph.nodes;
        printf("%d->%d | GPU=%f CPU=%f\n", row, col, h_projections[v],
               ref_projections[v]);
        num_errors += (int)(h_projections[v] != ref_projections[v]);
      }
    }
    if (num_errors == 0) {
      printf("======= PASSED ======\n");
    } else {
      printf("======= FAILED ======\n");
      util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }
  }

  return num_errors;
}

}  // namespace proj
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
