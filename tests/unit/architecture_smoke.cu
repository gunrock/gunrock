#include <gunrock/compat/runtime_api.h>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace {

__global__ void write_indices(int* values, int size) {
  auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < size) {
    values[index] = index + 1;
  }
}

}  // namespace

TEST(architecture, device_kernel_and_runtime) {
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GE(device_count, 1);

  constexpr int size = 8;
  thrust::device_vector<int> device_values(size, 0);
  write_indices<<<1, size>>>(device_values.data().get(), size);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  thrust::host_vector<int> host_values(device_values);
  for (int index = 0; index < size; ++index) {
    EXPECT_EQ(host_values[index], index + 1);
  }
}
