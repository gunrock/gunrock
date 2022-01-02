#pragma once

#include <chrono>
#include <vector>
#include <queue>

namespace ppr_cpu {

using namespace std;
using namespace std::chrono;

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
float run(csr_t& csr,
          vertex_t& n_seeds,
          weight_t* all_p,
          weight_t& alpha,
          weight_t& epsilon) {
  thrust::host_vector<edge_t> _rowptr(csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> _columns(csr.column_indices);
  thrust::host_vector<weight_t> _csr_data(csr.nonzero_values);

  edge_t* rowptr = _rowptr.data();
  vertex_t* columns = _columns.data();
  weight_t* csr_data = _csr_data.data();

  vertex_t n_nodes = csr.number_of_rows;

  auto t_start = high_resolution_clock::now();

  std::vector<weight_t> r(n_nodes);
  std::vector<weight_t> r_prime(n_nodes);
  std::vector<vertex_t> f(n_nodes);
  std::vector<vertex_t> f_prime(n_nodes);
  std::vector<vertex_t> degrees(n_nodes);

  for (vertex_t seed = 0; seed < n_seeds; seed++) {
    weight_t* p = all_p + (seed * n_nodes);

    for (vertex_t i = 0; i < n_nodes; i++) {
      r[i] = 0;
      r_prime[i] = 0;
      f[i] = 0;
      f_prime[i] = 0;
      degrees[i] = rowptr[i + 1] - rowptr[i];
    }

    r[seed] = 1;
    r_prime[seed] = 1;
    f[0] = seed;

    vertex_t f_size = 1;
    vertex_t f_prime_size = 0;

    while (f_size > 0) {
      for (vertex_t i = 0; i < f_size; i++) {
        vertex_t node_idx = f[i];
        p[node_idx] += (2 * alpha) / (1 + alpha) * r[node_idx];
        r_prime[node_idx] = 0;
      }

      for (vertex_t i = 0; i < f_size; i++) {
        vertex_t src_idx = f[i];
        vertex_t deg = degrees[src_idx];
        vertex_t offset = rowptr[src_idx];
        weight_t inv_r_deg = r[src_idx] / deg;

        for (vertex_t j = 0; j < deg; j++) {
          vertex_t dst_idx = columns[offset + j];
          weight_t update = ((1 - alpha) / (1 + alpha)) * inv_r_deg;

          weight_t oldval = r_prime[dst_idx];
          weight_t newval = r_prime[dst_idx] + update;
          weight_t thresh = degrees[dst_idx] * epsilon;

          r_prime[dst_idx] = newval;

          if ((oldval < thresh) && (newval >= thresh)) {
            f_prime[f_prime_size] = dst_idx;
            f_prime_size++;
          }
        }
      }

      r_prime = r;
      f.swap(f_prime);
      f_size = f_prime_size;
      f_prime_size = 0;
    }
  }

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace ppr_cpu