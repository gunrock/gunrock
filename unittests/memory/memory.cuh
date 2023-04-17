#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/error.hxx>  // error checking
#include <gunrock/memory.hxx>

template <std::size_t N>
__global__ void kernel(int* a) {
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx >= N)
    return;
  a[idx] = idx;
  printf("a[%i] = %i\n", idx, a[idx]);
}

void test_memory() {
  using namespace gunrock;
  using T = int;

  error::error_t status = cudaSuccess;
  memory::memory_space_t location = memory::memory_space_t::device;

  constexpr std::size_t N = 10;
  constexpr std::size_t bytes = N * sizeof(T);

  // cudaMalloc(&device_ptr, bytes);
  T* device_ptr = memory::allocate<T>(bytes, location);

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);

  kernel<N><<<1, N>>>(device_ptr);

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);

  memory::free(device_ptr, location);

  // Let's cause an exception
  // memory::free(device_ptr, location);  // free again
}

int main(int argc, char** argv) {
  test_memory();
  return EXIT_SUCCESS;
}