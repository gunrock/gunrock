
#pragma once

#include <chrono>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <gunrock/algorithms/generate/random.hxx>

namespace mst_cpu {

using namespace std;
using namespace std::chrono;

using vertex_t = int;
using edge_t = int;
using weight_t = float;

struct super {
  super* root;
  weight_t min_weight;
  vertex_t min_neighbor;
};

// Pointer jumping to form super vertices
void jump_pointers(super* supers, vertex_t v) {
  super* u = supers[v].root;
  while (u->root != u) {
    u = u->root;
  }
  supers[v].root = u;
  return;
}

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float run(csr_t& csr, weight_t* mst_weight) {
  // Copy data to CPU
  thrust::host_vector<edge_t> row_offsets(csr.row_offsets);
  thrust::host_vector<vertex_t> column_indices(csr.column_indices);
  thrust::host_vector<weight_t> nonzero_values(csr.nonzero_values);

  auto t_start = high_resolution_clock::now();

  vertex_t n_vertices = csr.number_of_rows;

  struct super* supers = new super[n_vertices];
  int super_vertices = n_vertices;

  *mst_weight = 0;

  // Initialize: set the root for each vertex to itself
  for (vertex_t i = 0; i < n_vertices; i++) {
    supers[i].root = &supers[i];
    supers[i].min_neighbor = i;
  }
  int mst_edges = 0;
  int it = 0;

  while (super_vertices > 1) {
    int start_super_vertices = super_vertices;
    it++;

    // Reset minimum weights and neighbors on each iteration
    for (vertex_t i = 0; i < n_vertices; i++) {
      supers[i].min_weight = std::numeric_limits<weight_t>::max();
      supers[i].min_neighbor = i;
    }

    // Iterate over each edge and compare to find the minimum weight and
    // neighbor for each super vertex
    for (vertex_t v = 0; v < n_vertices; v++) {
      edge_t start_edge = row_offsets[v];
      edge_t num_neighbors = row_offsets[v + 1] - row_offsets[v];

      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        vertex_t u = column_indices[e];
        // Do not check if they are already part of the same super vertex or if
        // v is less than u (we don't need to check duplicate / reverse edges)
        if (supers[v].root == supers[u].root || v < u) {
          continue;
        }
        weight_t new_dist = nonzero_values[e];
        if (new_dist < supers[v].root->min_weight) {
          supers[v].root->min_weight = new_dist;
          supers[v].root->min_neighbor = u;
        }
        if (new_dist < supers[u].root->min_weight) {
          supers[u].root->min_weight = new_dist;
          supers[u].root->min_neighbor = v;
        }
      }
    }

    // Add the minimum weight edges to the MST
    for (vertex_t v = 0; v < n_vertices; v++) {
      if (supers[v].min_weight != std::numeric_limits<weight_t>::max()) {
        // Jump pointers for the relevant vertices because their roots may have
        // been updated to be the same
        jump_pointers(supers, v);
        jump_pointers(supers, supers[v].min_neighbor);
        // To prevent duplicate edges, check that either
        // the source vertex index is
        // less than the destination vertex index or that the edge with the
        // source and destination flipped is not included.
        if (supers[supers[v].min_neighbor].root != supers[v].root) {
          if (supers[supers[v].min_neighbor].min_neighbor != v ||
              v < supers[v].min_neighbor) {
            *mst_weight += supers[v].min_weight;
            mst_edges++;
            supers[v].root->root = supers[supers[v].min_neighbor].root;
            super_vertices--;
          }
        }
      }
    }

    if (start_super_vertices == super_vertices) {
      printf("Error: invalid graph (super vertices not decremented)\n");
      exit(1);
    }

    // Update the root of each vertex. When adding an edge to the MST, we
    // update the source's root's root to the destination's root.
    // However, a vertex that had the source's root as its root may not
    // be updated. We must jump from root to root until the current vertex
    // and root are equal to find the new roots.
    for (vertex_t v = 0; v < n_vertices; v++) {
      jump_pointers(supers, v);
    }
  }

  delete (supers);

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace mst_cpu