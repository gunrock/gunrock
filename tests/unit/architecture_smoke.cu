#include <gunrock/compat/runtime_api.h>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <array>

namespace {

__global__ void write_indices(int* values, int size, int* completed) {
  auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < size) {
    values[index] = index + 1;
    atomicAdd(completed, 1);
  }
}

}  // namespace

TEST(architecture, device_kernel_and_runtime) {
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GE(device_count, 1);

  constexpr int size = 65;
  thrust::device_vector<int> device_values(size, 0);
  thrust::device_vector<int> completed(1, 0);
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;
  ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
  ASSERT_EQ(hipEventCreate(&event), hipSuccess);

  write_indices<<<2, 64, 0, stream>>>(device_values.data().get(), size,
                                      completed.data().get());
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipEventRecord(event, stream), hipSuccess);
  ASSERT_EQ(hipEventSynchronize(event), hipSuccess);

  std::array<int, size> host_values{};
  ASSERT_EQ(hipMemcpyAsync(host_values.data(), device_values.data().get(),
                           sizeof(host_values), hipMemcpyDeviceToHost, stream),
            hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
  for (int index = 0; index < size; ++index) {
    EXPECT_EQ(host_values[index], index + 1);
  }
  thrust::host_vector<int> host_completed(completed);
  ASSERT_EQ(host_completed[0], size);

  EXPECT_EQ(hipEventDestroy(event), hipSuccess);
  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}
