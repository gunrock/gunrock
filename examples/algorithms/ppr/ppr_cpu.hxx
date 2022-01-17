#pragma once

#include <chrono>
#include <vector>
#include <queue>

#include <gunrock/formats/formats.hxx>
#include <gunrock/memory.hxx>

namespace ppr_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float run(csr_t& csr,
          vertex_t& n_seeds,
          weight_t* all_p,
          weight_t& alpha,
          weight_t& epsilon) {
  using namespace gunrock;
  using namespace memory;
  format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> h_csr(csr);
  vertex_t n_nodes = h_csr.number_of_rows;

  auto t_start = high_resolution_clock::now();

  std::vector<weight_t> r(n_nodes);
  std::vector<weight_t> r_prime(n_nodes);

  // Input and output frontiers:
  std::vector<vertex_t> f;
  std::vector<vertex_t> f_prime;

  std::vector<vertex_t> degrees(n_nodes);

  // Calculate degrees:
  for (vertex_t i = 0; i < n_nodes; i++)
    degrees[i] = h_csr.row_offsets[i + 1] - h_csr.row_offsets[i];

  // Batched over n_seeds.
  for (vertex_t seed = 0; seed < n_seeds; seed++) {
    weight_t* p = all_p + (seed * n_nodes);

    // Set seed values and frontier seed active:
    r[seed] = 1;
    r_prime[seed] = 1;
    f.push_back(seed);

    while (f.size() > 0) {
      // For all elements in the frontier.
      for (vertex_t i = 0; i < f.size(); i++) {
        vertex_t node_idx = f[i];
        p[node_idx] += (2 * alpha) / (1 + alpha) * r[node_idx];
        r_prime[node_idx] = 0;
      }

      // For all elements in the frontier.
      for (vertex_t i = 0; i < f.size(); i++) {
        vertex_t src_idx = f[i];
        weight_t inv_r_deg = r[src_idx] / degrees[src_idx];

        // For all neighbors.
        for (vertex_t j = h_csr.row_offsets[src_idx];
             j < h_csr.row_offsets[src_idx + 1]; j++) {
          vertex_t dst_idx = h_csr.column_indices[j];
          weight_t update = ((1 - alpha) / (1 + alpha)) * inv_r_deg;

          weight_t oldval = r_prime[dst_idx];
          weight_t newval = r_prime[dst_idx] + update;
          weight_t thresh = degrees[dst_idx] * epsilon;

          r_prime[dst_idx] = newval;

          if ((oldval < thresh) && (newval >= thresh)) {
            f_prime.push_back(dst_idx);
          }
        }
      }

      r = r_prime;
      f.swap(f_prime);
      f_prime.clear();
    }

    // Reset the data for next seed:
    f.clear();
    f_prime.clear();

    // Reset r and r_prime:
    std::fill(r.begin(), r.end(), 0);
    std::fill(r_prime.begin(), r_prime.end(), 0);
  }

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace ppr_cpu