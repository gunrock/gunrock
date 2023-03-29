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

  // shared pointer with a custom deleter
  auto deleter = [&](T* ptr) { memory::free(ptr, location); };
  std::shared_ptr<T> device_ptr(memory::allocate<T>(bytes, location), deleter);

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);

  kernel<N><<<1, N>>>(device_ptr.get());

  status = cudaDeviceSynchronize();
  if (cudaSuccess != status)
    throw error::exception_t(status);

  // no need to free the memory, it is automatically
  // freed on completion. no memory leaks.
}

int main(int argc, char** argv) {
  test_memory();
  return EXIT_SUCCESS;
}