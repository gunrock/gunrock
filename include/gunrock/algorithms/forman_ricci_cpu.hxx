#pragma once

#include <vector>

namespace gunrock {
namespace forman_ricci {
namespace cpu {

// =============================================================================
// Community detection using BFS (threshold-based) - CPU version
// =============================================================================
template <typename vertex_t, typename edge_t, typename weight_t>
int find_communities_cpu(
    vertex_t* h_row_offsets,
    vertex_t* h_column_indices,
    weight_t* h_edge_weights,
    weight_t threshold,
    int* h_component_ids,
    vertex_t n_vertices) {
  
  for (vertex_t i = 0; i < n_vertices; i++) {
    h_component_ids[i] = -1;
  }
  
  std::vector<vertex_t> queue;
  int n_components = 0;
  
  for (vertex_t start = 0; start < n_vertices; start++) {
    if (h_component_ids[start] != -1) continue;
    
    queue.clear();
    queue.push_back(start);
    h_component_ids[start] = n_components;
    size_t queue_idx = 0;
    
    while (queue_idx < queue.size()) {
      vertex_t node = queue[queue_idx++];
      
      for (edge_t edge = h_row_offsets[node]; edge < h_row_offsets[node + 1]; edge++) {
        vertex_t neighbor = h_column_indices[edge];
        weight_t w = h_edge_weights[edge];
        
        if (w < threshold && h_component_ids[neighbor] == -1) {
          h_component_ids[neighbor] = n_components;
          queue.push_back(neighbor);
        }
      }
    }
    n_components++;
  }
  
  return n_components;
}

// =============================================================================
// Modularity calculation - CPU version
// =============================================================================
template <typename vertex_t, typename edge_t>
double calculate_modularity_cpu(
    vertex_t* h_row_offsets,
    vertex_t* h_column_indices,
    int* h_component_ids,
    vertex_t n_vertices,
    edge_t n_undirected_edges) {
  
  double modularity = 0.0;
  edge_t m = n_undirected_edges;
  
  for (vertex_t u = 0; u < n_vertices; u++) {
    int k_u = h_row_offsets[u + 1] - h_row_offsets[u];
    
    for (edge_t edge = h_row_offsets[u]; edge < h_row_offsets[u + 1]; edge++) {
      vertex_t v = h_column_indices[edge];
      int k_v = h_row_offsets[v + 1] - h_row_offsets[v];
      
      if (h_component_ids[u] == h_component_ids[v]) {
        modularity += 1.0 - (double)(k_u * k_v) / (2.0 * m);
      } else {
        modularity += 0.0 - (double)(k_u * k_v) / (2.0 * m);
      }
    }
  }
  
  return modularity / (2.0 * m);
}


}  // namespace cpu
}  // namespace forman_ricci
}  // namespace gunrock