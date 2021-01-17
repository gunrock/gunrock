/**
 * @brief GEO test for shared library CXX interface
 * @file shared_lib_hits.cu
 */

 #include <stdio.h>

#include <cmath>

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

  std::vector<float> latitudes{ 37.7449063493, NAN, 33.9774659389, NAN, 39.2598884729  };
  std::vector<float> longitudes{-122.009432884, NAN, -114.886512278, NAN, -106.804662071 };

  thrust::device_vector<float> v_latitudes(latitudes);
  thrust::device_vector<float> v_longitudes(longitudes);

  float* d_lat = v_latitudes.data().get();
  float* d_long = v_longitudes.data().get();

  //
  //  Host call
  //
  std::cout << "host memory call" << std::endl;
  geo<int, int, float>(num_verts, num_edges, graph_offsets.data(), graph_indices.data(), 2, /* geo_iter */ 1, 
  latitudes.data(), longitudes.data(), gunrock::util::HOST);

  printf("lat,long\n");
  for (int i = 0; i < num_verts; i++) {
    printf("Node_ID: [%d], Latitude: [%6f], Longitude: [%6f]\n", i, latitudes[i], longitudes[i]);
  }

  //
  //  Device call
  //
  std::cout << "device memory call" << std::endl;

  geo<int, int, float>(num_verts, num_edges, graph_offsets_d.data().get(), graph_indices_d.data().get(), 2, 1,
    v_latitudes.data().get(), v_longitudes.data().get(),
    gunrock::util::DEVICE);

  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(1),
                   [num_verts, d_lat, d_long] __device__ (int) {

                      printf("lat,long\n");
                      for (int i = 0; i < num_verts; i++) {
                        printf("Node_ID: [%d], Latitude: [%6f], Longitude: [%6f]\n", i, d_lat[i], d_long[i]);
                      }
                   });

}
