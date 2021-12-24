/**
 * @file array.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Unit test for array container.
 * @version 0.1
 * @date 2021-12-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <gunrock/error.hxx>
#include <gunrock/container/array.hxx>

template <std::size_t N>
__global__ void kernel() {
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx >= N)
    return;

  gunrock::array<float, N> a;
  a[idx] = (float)idx;
}

TEST(containers, array) {
  using namespace gunrock;

  error::error_t status = cudaSuccess;

  constexpr std::size_t N = 10;
  gunrock::array<float, N> a;
  float* pointer = a.data();
  const float* const_pointer = a.data();
  std::size_t size = a.size();
  std::size_t max_size = a.max_size();
  bool is_empty = a.empty();

  ASSERT_EQ(is_empty, false);
  ASSERT_EQ(size, N);
  ASSERT_EQ(max_size, N);

  for (std::size_t i = 0; i < N; ++i) {
    a[i] = (float)i;
  }

  status = cudaDeviceSynchronize();
  ASSERT_EQ(status, cudaSuccess)
      << "CUDART error: " << cudaGetErrorString(status);

  kernel<N><<<1, N>>>();

  status = cudaDeviceSynchronize();
  ASSERT_EQ(status, cudaSuccess)
      << "CUDART error: " << cudaGetErrorString(status);
}