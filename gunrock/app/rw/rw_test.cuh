// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_test.cu
 *
 * @brief Test related functions for rw
 */

#pragma once

namespace gunrock {
namespace app {
namespace rw {

/******************************************************************************
 * TW Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference RW ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
 * @param[in]   walk_length   Length of random walks
 * @param[in]   walks         Array to store random walk in
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(const GraphT &graph, int walk_length, int walks_per_node,
                     int walk_mode, bool store_walks,
                     typename GraphT::VertexT *walks, bool quiet) {
  printf("CPU_Reference: start\n");

  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  if (store_walks) {
    for (SizeT i = 0; i < graph.nodes * walk_length * walks_per_node; ++i) {
      walks[i] = -1;
    }
  }

  uint64_t total_neighbors_seen = 0;
  uint64_t total_steps_taken = 0;

  if (walk_mode == 0) {  // Random
    // <TODO>
    printf("CPU_Reference: NotImplemented for walk_mode=0\n");
    // </TODO>
  } else if (walk_mode == 1) {  // Max
    for (int walk_id = 0; walk_id < graph.nodes * walks_per_node; walk_id++) {
      VertexT node_id = (VertexT)(walk_id % graph.nodes);
      for (int step = 0; step < walk_length; step++) {
        if (store_walks) {
          walks[walk_id * walk_length + step] = node_id;
        }

        if (step == walk_length - 1) break;

        SizeT num_neighbors = graph.GetNeighborListLength(node_id);
        if (num_neighbors == 0) break;

        SizeT neighbor_list_offset = graph.GetNeighborListOffset(node_id);

        VertexT max_neighbor_id = graph.GetEdgeDest(neighbor_list_offset);
        ValueT max_neighbor_val = graph.node_values[max_neighbor_id];

        for (SizeT offset = 1; offset < num_neighbors; offset++) {
          VertexT neighbor = graph.GetEdgeDest(neighbor_list_offset + offset);
          ValueT neighbor_val = graph.node_values[neighbor];
          if (neighbor_val > max_neighbor_val) {
            max_neighbor_id = neighbor;
            max_neighbor_val = neighbor_val;
          }
        }

        node_id = max_neighbor_id;
        total_neighbors_seen += num_neighbors;
        total_steps_taken++;
      }
    }
    printf("CPU_Reference: total_neighbors_seen = %d\n", total_neighbors_seen);
    printf("CPU_Reference: total_steps_taken    = %d\n", total_steps_taken);
  } else if (walk_mode == 2) {
    // <TODO>
    printf("CPU_Reference: NotImplemented for walk_mode=2\n");
    // </TODO>
  } else {
    printf("CPU_Reference: Unkown walk_mode=%d\n", walk_mode);
  }

  printf("CPU_Reference: done\n");
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  return elapsed;
}

/**
 * @brief Validation of RW results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  walk_length         Random walk length
 * @param[in]  walks_per_node      Number of random walks per node
 * @param[in]  h_walks       GPU walks
 * @param[in]  ref_walks     CPU walks
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph, int walk_length,
    int walks_per_node, int walk_mode, bool store_walks,
    typename GraphT::VertexT *h_walks, uint64_t *h_neighbors_seen,
    uint64_t *h_steps_taken, typename GraphT::VertexT *ref_walks) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  bool quiet_mode = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");

  uint64_t total_neighbors_seen = 0;
  uint64_t total_steps_taken = 0;
  for (SizeT v = 0; v < graph.nodes * walks_per_node; v++) {
    total_neighbors_seen += (uint64_t)h_neighbors_seen[v];
    total_steps_taken += (uint64_t)h_steps_taken[v];
  }
  printf("Validate_Results: total_neighbors_seen=%ld\n", total_neighbors_seen);
  printf("Validate_Results: total_steps_taken=%ld\n", total_steps_taken);

  SizeT num_errors = 0;
  if (!quick && store_walks) {
    if (!quiet_mode) printf("[[");

    for (SizeT v = 0; v < graph.nodes * walk_length * walks_per_node; ++v) {
      if (!quiet_mode) {
        if ((v > 0) && (v % walk_length == 0)) {
          printf("],\n[");
        }

        printf("%d:%d, ", h_walks[v], ref_walks[v]);
      }

      if (walk_mode == 1) {  // Only for greedy, because it's deterministic
        if (h_walks[v] != ref_walks[v]) {
          num_errors++;
        }
      }
    }

    if (!quiet_mode) printf("]]\n");
    printf("-------------------\n");
    printf("%d errors occurred.\n", num_errors);
  } else {
    printf("-------- NO VALIDATION --------\n");
  }

  return num_errors;
}

}  // namespace rw
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
