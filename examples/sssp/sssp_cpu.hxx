#ifndef __SSSP_CPU_H__
#define __SSSP_CPU_H__

#include <chrono>
#include <vector>
#include <queue>

namespace sssp_cpu {

  using namespace std;
  using namespace std::chrono;
  
  template <typename vertex_t, typename weight_t>
  class prioritize {
      public:
          bool operator()(pair<vertex_t, weight_t> &p1, pair<vertex_t, weight_t> &p2) {
              return p1.second > p2.second;
          }
  };

  template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>
  float run(
    csr_t& csr,
    vertex_t& single_source,
    weight_t* distances,
    vertex_t* predecessors
  ) { 
    
    thrust::host_vector<edge_t>   row_offsets(csr.row_offsets);       // Copy data to CPU
    thrust::host_vector<vertex_t> column_indices(csr.column_indices);
    thrust::host_vector<weight_t> nonzero_values(csr.nonzero_values);
    
    for(vertex_t i = 0; i < csr.number_of_rows; i++)
      distances[i] = std::numeric_limits<weight_t>::max();
    
    auto t_start = high_resolution_clock::now();
    
    distances[single_source] = 0;

    priority_queue<pair<vertex_t,weight_t>, std::vector<pair<vertex_t,weight_t>>, prioritize<vertex_t, weight_t>> pq;
    pq.push(make_pair(single_source, 0.0));

    while(!pq.empty()) {
        pair<vertex_t, weight_t> curr = pq.top();
        pq.pop();
        
        vertex_t curr_node = curr.first;
        weight_t curr_dist = curr.second;
        
        vertex_t start = row_offsets[curr_node];
        vertex_t end   = row_offsets[curr_node + 1];
        
        for(vertex_t offset = start; offset < end; offset++) {
            vertex_t neib     = column_indices[offset];
            weight_t new_dist = curr_dist + nonzero_values[offset];
            if(new_dist < distances[neib]) {
                distances[neib] = new_dist;
                pq.push(make_pair(neib, new_dist));
            }
        }
    }

    auto t_stop  = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
    return (float)elapsed / 1000;
  }

  template <typename val_t>
  int compute_error(
    thrust::device_vector<val_t> _gpu_result,
    thrust::host_vector<val_t> cpu_result
  ) {
    thrust::host_vector<val_t> gpu_result(_gpu_result);
    
    int n_errors = 0;
    for(int i = 0 ; i < cpu_result.size(); i++) {
      if(gpu_result[i] != cpu_result[i]) {
        n_errors++;
      }
    }
    return n_errors;
  }
  
}

#endif