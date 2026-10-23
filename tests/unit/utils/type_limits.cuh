#include <gunrock/util/type_limits.hxx>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <limits>

__global__ void check_device_limits(int* result) {
  result[0] = gunrock::util::limits::is_valid(7) ? 1 : 0;
  result[1] =
      gunrock::util::limits::is_valid(gunrock::numeric_limits<int>::invalid())
          ? 1
          : 0;
  result[2] = gunrock::util::limits::is_valid(1.25f) ? 1 : 0;
  result[3] =
      gunrock::util::limits::is_valid(gunrock::numeric_limits<float>::invalid())
          ? 1
          : 0;
}

TEST(utils, type_limits_identify_invalid_sentinels) {
  EXPECT_EQ(gunrock::numeric_limits<int>::invalid(), -1);
  EXPECT_EQ(gunrock::numeric_limits<unsigned int>::invalid(),
            std::numeric_limits<unsigned int>::max());
  EXPECT_TRUE(std::isnan(gunrock::numeric_limits<float>::invalid()));
  EXPECT_TRUE(std::isnan(gunrock::numeric_limits<double>::invalid()));

  EXPECT_TRUE(gunrock::util::limits::is_valid(0));
  EXPECT_FALSE(gunrock::util::limits::is_valid(-1));
  EXPECT_TRUE(gunrock::util::limits::is_valid(0.0f));
  EXPECT_FALSE(gunrock::util::limits::is_valid(
      gunrock::numeric_limits<float>::invalid()));
}

TEST(utils, type_limits_work_in_device_code) {
  thrust::device_vector<int> result(4, -1);
  check_device_limits<<<1, 1>>>(result.data().get());
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  thrust::host_vector<int> host_result(result);
  expect_values(host_result, {1, 0, 1, 0});
}
