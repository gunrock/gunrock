#include <gunrock/error.hxx>  // error checking
#include <gunrock/memory.hxx>
#include <gunrock/compat/runtime_api.h>

#include <gtest/gtest.h>

#include <array>

template <std::size_t N>
__global__ void write_sequence(int* values) {
  int idx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (idx >= N)
    return;
  values[idx] = idx * idx;
}

TEST(memory, device_allocation_round_trips_values) {
  constexpr std::size_t N = 10;
  constexpr std::size_t bytes = N * sizeof(int);
  auto* device_values = gunrock::memory::allocate<int>(bytes);
  ASSERT_NE(device_values, nullptr);

  write_sequence<N><<<1, N>>>(device_values);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  std::array<int, N> host_values{};
  ASSERT_EQ(hipMemcpy(host_values.data(), device_values, bytes,
                      hipMemcpyDeviceToHost),
            hipSuccess);
  for (std::size_t index = 0; index < N; ++index) {
    EXPECT_EQ(host_values[index], static_cast<int>(index * index));
  }

  gunrock::memory::free(device_values);
  EXPECT_EQ(gunrock::memory::allocate<int>(0), nullptr);
}