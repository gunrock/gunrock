#include <gunrock/error.hxx>  // error checking
#include <gunrock/memory.hxx>
#include <gunrock/compat/runtime_api.h>

#include <gtest/gtest.h>

template <std::size_t N>
__global__ void kernel(int* a) {
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx >= N)
    return;
  a[idx] = idx;
  printf("a[%i] = %i\n", idx, a[idx]);
}

TEST(memory, memory) {
  using namespace gunrock;
  using T = int;

  error::error_t status = hipSuccess;
  memory::memory_space_t location = memory::memory_space_t::device;

  constexpr std::size_t N = 10;
  constexpr std::size_t bytes = N * sizeof(T);

  T* device_ptr = memory::allocate<T>(bytes, location);

  status = hipDeviceSynchronize();
  if (hipSuccess != status)
    throw error::exception_t(status);

  kernel<N><<<1, N>>>(device_ptr);

  status = hipDeviceSynchronize();
  if (hipSuccess != status)
    throw error::exception_t(status);

  memory::free(device_ptr, location);
}