#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/forman_ricci.hxx>
#include <gunrock/algorithms/forman_ricci_cpu.hxx>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <vector>
#include <iomanip>
#include <algorithm>

using namespace gunrock;
using namespace memory;


// =============================================================================
// Main test function
// =============================================================================
void test_forman_ricci(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  using vertex_t = int;
  using edge_t = int;
  using weight_t = double;

  std::string filename = argument_array[1];

  // ==========================================================================
  // Load graph - keep COO before converting to CSR
  // ==========================================================================
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  // Convert to CSR
  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(coo);

  vertex_t n_verts = csr.row_offsets.size() - 1;
  thrust::host_vector<vertex_t> h_offsets = csr.row_offsets;
  for (vertex_t i = 0; i < n_verts; i++) {
      thrust::sort(thrust::device,
        csr.column_indices.begin() + h_offsets[i],
        csr.column_indices.begin() + h_offsets[i + 1]);
  }

  // Build graph
  auto G = graph::build(properties, csr);
  vertex_t n_vertices = G.get_number_of_vertices();
  edge_t n_total_edges = G.get_number_of_edges();
  edge_t n_undirected_edges = n_total_edges / 2;

  std::cout << "=== Graph Info ===" << std::endl;
  std::cout << "Vertices: " << n_vertices << std::endl;
  std::cout << "Undirected edges: " << n_undirected_edges << std::endl;
  std::cout << "Total edges (CSR): " << n_total_edges << std::endl;

  // ==========================================================================
  // Allocate edge weights
  // ==========================================================================
  thrust::device_vector<weight_t> d_edge_weights(n_total_edges);
  
  // ==========================================================================
  // Run Forman-Ricci flow using Gunrock pattern
  // ==========================================================================
  int n_iterations = 10;

  // std::cout << "\n=== Running Forman-Ricci Flow ===" << std::endl;
  std::cout << "\n=== Running Forman-Ricci Flow (" << n_iterations << " iterations) ===" << std::endl;
 
  float gpu_time = gunrock::forman_ricci::run(
    G, n_iterations, d_edge_weights.data().get());

  std::cout << "\nRicci flow time: " << gpu_time << " ms" << std::endl;

  // ==========================================================================
  // Copy results to host for analysis
  // ==========================================================================
  thrust::host_vector<weight_t> h_edge_weights = d_edge_weights;
  thrust::host_vector<vertex_t> h_row_offsets = csr.row_offsets;
  thrust::host_vector<vertex_t> h_column_indices = csr.column_indices;

  auto min_it = std::min_element(h_edge_weights.begin(), h_edge_weights.end());
  auto max_it = std::max_element(h_edge_weights.begin(), h_edge_weights.end());
  weight_t min_w = *min_it;
  weight_t max_w = *max_it;
  
  std::cout << "\n=== Weight Analysis ===" << std::endl;
  std::cout << "Min weight: " << min_w << std::endl;
  std::cout << "Max weight: " << max_w << std::endl;
  std::cout << "Ratio: " << max_w / min_w << std::endl;

  // ==========================================================================
  // Threshold search for community detection (using original quantile approach)
  // ==========================================================================
  
  // Paper parameters
  constexpr float QUANTILE_Q = 0.999f;
  constexpr float DELTA_STEP = 0.25f;
  
  thrust::device_vector<weight_t> d_sorted_weights = d_edge_weights;
  thrust::sort(d_sorted_weights.begin(), d_sorted_weights.end());
  thrust::host_vector<weight_t> h_sorted_weights = d_sorted_weights;
  
  std::cout << "\n=== Community Detection ===" << std::endl;
  
  // Get 99.9th quantile
  int q_idx = (int)(QUANTILE_Q * n_total_edges);
  if (q_idx >= n_total_edges) q_idx = n_total_edges - 1;
  weight_t w_quantile = h_sorted_weights[q_idx];
  weight_t w_stop = 1.1f * min_w;
  
  printf("Weight quantile (%.3f): %.6f\n", QUANTILE_Q, w_quantile);
  
  // Build cutoff list
  std::vector<weight_t> cutoff_list;
  
  // Cutoffs above quantile (actual edge weights, skip duplicates)
  for (int idx = 0; idx < n_total_edges; idx+=2) {
    if (h_sorted_weights[idx] > w_quantile) {
      if (cutoff_list.empty() || h_sorted_weights[idx] != cutoff_list.back()) {
        cutoff_list.push_back(h_sorted_weights[idx]);
      }
    }
  }
  
  // Uniform steps below quantile
  for (weight_t t = w_quantile - DELTA_STEP; t >= w_stop; t -= DELTA_STEP) {
    cutoff_list.push_back(t);
  }
  
  int n_cutoff = cutoff_list.size();
  printf("Number of cutoff thresholds: %d\n\n", n_cutoff);

  std::vector<int> h_component_ids(n_vertices);
  
  double best_modularity = -1.0;
  weight_t best_threshold = 0.0;
  int best_n_communities = 0;
  std::vector<int> best_labels(n_vertices);
  
  std::cout << "Threshold       Communities  Modularity" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  
  for (int cutoff_idx = 0; cutoff_idx < n_cutoff; cutoff_idx++) {
    weight_t threshold = cutoff_list[cutoff_idx];
    
    int n_communities = gunrock::forman_ricci::cpu::find_communities_cpu<vertex_t, edge_t, weight_t>(
        h_row_offsets.data(),
        h_column_indices.data(),
        h_edge_weights.data(),
        threshold,
        h_component_ids.data(),
        n_vertices);    
    
    if (n_communities < 2 || n_communities > n_vertices / 2) continue;
    
    double modularity = gunrock::forman_ricci::cpu::calculate_modularity_cpu<vertex_t, edge_t>(
        h_row_offsets.data(),
        h_column_indices.data(),
        h_component_ids.data(),
        n_vertices,
        n_undirected_edges);
    
    if (n_communities <= 30) {
      char marker = (modularity > best_modularity) ? '*' : ' ';
      printf("%.6e  %-12d %-12.6f %c\n", 
             threshold, n_communities, modularity, marker);
    }
    
    if (modularity > best_modularity) {
      best_modularity = modularity;
      best_threshold = threshold;
      best_n_communities = n_communities;
      best_labels = h_component_ids;
    }
  }

  // ==========================================================================
  // Final results with community sizes
  // ==========================================================================
  
  // Rerun with best threshold
  gunrock::forman_ricci::cpu::find_communities_cpu<vertex_t, edge_t, weight_t>(
      h_row_offsets.data(),
      h_column_indices.data(),
      h_edge_weights.data(),
      best_threshold,
      best_labels.data(),
      n_vertices);

  std::vector<int> community_sizes(best_n_communities, 0);
  for (vertex_t i = 0; i < n_vertices; i++) {
    if (best_labels[i] >= 0 && best_labels[i] < best_n_communities) {
      community_sizes[best_labels[i]]++;
    }
  }
  
  std::vector<std::pair<int, int>> size_pairs;
  for (int c = 0; c < best_n_communities; c++) {
    size_pairs.push_back({community_sizes[c], c});
  }
  std::sort(size_pairs.begin(), size_pairs.end(), std::greater<std::pair<int, int>>());
  
  int largest_community_size = size_pairs[0].first;
  int smallest_community_size = size_pairs[best_n_communities - 1].first;
  bool is_trivial = (largest_community_size > n_vertices * 0.95) || 
                    (smallest_community_size < 10);
  
  std::cout << "\n=== FINAL RESULTS ===" << std::endl;
  std::cout << "Best threshold: " << best_threshold << std::endl;
  std::cout << "Communities found: " << best_n_communities << std::endl;
  std::cout << "Modularity: " << best_modularity << std::endl;
  std::cout << "Total GPU time: " << gpu_time << " ms" << std::endl;
  
  if (is_trivial) {
    std::cout << "\n*** WARNING: Trivial partition detected! ***" << std::endl;
  }
  
  std::cout << "\nCommunity sizes (sorted by size):" << std::endl;
  for (int i = 0; i < best_n_communities && i < 20; i++) {
    std::cout << "  Community " << size_pairs[i].second 
              << ": " << size_pairs[i].first << " nodes";
    if (size_pairs[i].first < 10) {
      std::cout << " (tiny!)";
    }
    std::cout << std::endl;
  }
  if (best_n_communities > 20) {
    std::cout << "  ... (" << (best_n_communities - 20) << " more communities)" << std::endl;
  }
}

int main(int argc, char** argv) {
  test_forman_ricci(argc, argv);
  return 0;
}
