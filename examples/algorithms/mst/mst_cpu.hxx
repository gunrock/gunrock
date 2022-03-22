
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

// pointer jumping to form super vertices
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

  edge_t* mst = new edge_t[nonzero_values.size()];

  *mst_weight = 0;

  for (vertex_t i = 0; i < n_vertices; i++) {
    supers[i].root = &supers[i];
    supers[i].min_neighbor = i;
  }
  int mst_edges = 0;
  int it = 0;
  while (super_vertices > 1) {
    it++;
    for (vertex_t i = 0; i < n_vertices; i++) {
      supers[i].min_weight = std::numeric_limits<weight_t>::max();
      supers[i].min_neighbor = i;
    }

    for (vertex_t v = 0; v < n_vertices; v++) {
      edge_t start_edge = row_offsets[v];
      edge_t num_neighbors = row_offsets[v + 1] - row_offsets[v];

      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        vertex_t u = column_indices[e];
        if (supers[v].root == supers[u].root) {
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

    for (vertex_t v = 0; v < n_vertices; v++) {
      if (supers[v].min_weight != std::numeric_limits<weight_t>::max()) {
        jump_pointers(supers, v);
        jump_pointers(supers, supers[v].min_neighbor);
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
    // add stop condition when super vertices not decremented

    for (vertex_t v = 0; v < n_vertices; v++) {
      jump_pointers(supers, v);
    }

    // need to free
  }

  // std::cout << "mst edges " << mst_edges << "\n";

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace mst_cpu