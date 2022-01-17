#pragma once

#include <chrono>
#include <vector>
#include <queue>

namespace bfs_cpu {

using namespace std;
using namespace std::chrono;

template <typename vertex_t>
class prioritize {
 public:
  bool operator()(pair<vertex_t, vertex_t>& p1, pair<vertex_t, vertex_t>& p2) {
    return p1.second > p2.second;
  }
};

template <typename csr_t, typename vertex_t, typename edge_t>
float run(csr_t& csr, vertex_t& single_source, vertex_t* distances) {
  thrust::host_vector<edge_t> _row_offsets(
      csr.row_offsets);  // Copy data to CPU
  thrust::host_vector<vertex_t> _column_indices(csr.column_indices);

  edge_t* row_offsets = _row_offsets.data();
  vertex_t* column_indices = _column_indices.data();

  for (vertex_t i = 0; i < csr.number_of_rows; i++)
    distances[i] = std::numeric_limits<vertex_t>::max();

  auto t_start = high_resolution_clock::now();

  distances[single_source] = 0;

  priority_queue<pair<vertex_t, vertex_t>,
                 std::vector<pair<vertex_t, vertex_t>>, prioritize<vertex_t>>
      pq;

  pq.push(make_pair(single_source, 0.0));

  while (!pq.empty()) {
    pair<vertex_t, vertex_t> curr = pq.top();
    pq.pop();

    vertex_t curr_node = curr.first;
    vertex_t curr_dist = curr.second;

    vertex_t start = row_offsets[curr_node];
    vertex_t end = row_offsets[curr_node + 1];

    for (vertex_t offset = start; offset < end; offset++) {
      vertex_t neib = column_indices[offset];
      vertex_t new_dist = curr_dist + 1;
      if (new_dist < distances[neib]) {
        distances[neib] = new_dist;
        pq.push(make_pair(neib, new_dist));
      }
    }
  }

  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace bfs_cpu