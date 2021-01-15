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

  std::vector<int> colors(num_verts);
  thrust::device_vector<int> v_colors(num_verts);

  int* d_colors = v_colors.data().get();

  //
  //  Host call
  //
  printf("host memory call\n");
  color<int, int, int>(num_verts, num_edges, graph_offsets.data(), graph_indices.data(), colors.data(), gunrock::util::HOST);

  printf("colors:\n");
  for (auto& i : colors) {
    printf("%d", i);
    printf(" ");
  }
  printf("\n");

  //
  //  Device call
  //
  printf("device memory call\n");

  color<int, int, int>(num_verts, num_edges, graph_offsets_d.data().get(), graph_indices_d.data().get(), v_colors.data().get(), gunrock::util::DEVICE);

  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [num_verts, d_colors] __device__ (int) {
                     printf("colors:\n");
                     for (int i = 0 ; i < num_verts ; ++i) {
                      printf("%d", d_colors[i]);
                      printf(" "); 
                     }
                    printf("\n");
                   });

}
