/**
 * @brief HITS test for shared library advanced interface
 * @file shared_lib_hits.cu
 */

#include <stdio.h>
#include <gunrock/gunrock.hpp>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

int main(int argc, char** argv) {

  std::vector<int>  graph_offsets{ 0, 2, 6, 7, 9, 10 };
  std::vector<int>  graph_indices{ 1, 3, 0, 2, 3, 4, 1, 0, 1, 1 };

  int num_verts = graph_offsets.size() - 1;
  int num_edges = graph_indices.size();

  thrust::device_vector<int>  graph_offsets_d(graph_offsets);
  thrust::device_vector<int>  graph_indices_d(graph_indices);

  std::vector<float> hub_ranks(num_verts);
  std::vector<float> auth_ranks(num_verts);
  
  thrust::device_vector<float> hub_ranks_v(num_verts);
  thrust::device_vector<float> auth_ranks_v(num_verts);

  float *d_hub_ranks  = hub_ranks_v.data().get();
  float *d_auth_ranks  = auth_ranks_v.data().get();

  int max_iter = 1000;
  float tol = 0.0001;
  int hits_norm = HITS_NORMALIZATION_METHOD_1;

  int HOST = 1;
  int DEVICE = 2;

  //
  //  Host call
  //
  printf("host memory call\n");
  hits<int, int, float>(num_verts, num_edges, graph_offsets.data(), graph_indices.data(), max_iter, tol, hits_norm, hub_ranks.data(), auth_ranks.data(), HOST);

  for (int i = 0 ; i < num_verts ; ++i) {
    printf("Node_ID: [%d], Hub Score: [%f], Auth Score: [%f]\n", i, hub_ranks[i], auth_ranks[i]);
  }
  //
  //  Device call
  //
  printf("device memory call\n");
  hits(num_verts, num_edges, graph_offsets_d.data().get(), graph_indices_d.data().get(), max_iter, tol, hits_norm, d_hub_ranks, d_auth_ranks, DEVICE);

  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [num_verts, d_hub_ranks, d_auth_ranks] __device__ (int) {
                     for (int i = 0 ; i < num_verts ; ++i)
                      //  printf("  [%d]: (%g, %g)\n", i, d_hub_ranks[i], d_auth_ranks[i]);
                      printf("Node_ID: [%d], Hub Score: [%f], Auth Score: [%f]\n", i, d_hub_ranks[i], d_auth_ranks[i]);
                   });
}
