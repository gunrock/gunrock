#include <cstdlib>  // EXIT_SUCCESS

#include <iostream>

#include <gunrock/container/array.hxx>
#include <gunrock/error.hxx>  // error checking

template <std::size_t N>
__global__ void kernel(int dummy) {
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx >= N)
    return;

  gunrock::array<float, N> a;
  a[idx] = (float)idx;
  printf("device[%i] = %f\n", idx, a[idx]);
}

void test_array() {
  using namespace gunrock;

  error::error_t status = cudaSuccess;

  constexpr std::size_t N = 10;

  gunrock::array<float, N> a;

  float* pointer = a.data();
  const float* const_pointer = a.data();

  std::size_t size = a.size();
  std::size_t max_size = a.max_size();
  bool is_empty = a.empty();

  std::cout << "Array.size() = " << size << std::endl;
  std::cout << "Array.max_size() = " << max_size << std::endl;
  std::cout << "Is Array Empty? " << std::boolalpha << is_empty << std::endl;

  for (std::size_t i = 0; i < N; ++i) {
    a[i] = i;
    std::cout << "host[" << i << "] = " << a[i] << std::endl;
  }

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);

  int dummy = 0;
  kernel<N><<<1, N>>>(dummy);

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);
}

int main(int argc, char** argv) {
  test_array();
  return EXIT_SUCCESS;
}