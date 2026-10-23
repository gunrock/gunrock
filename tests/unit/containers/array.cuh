/**
 * @file array.cuh
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Unit test for array container.
 * @date 2021-12-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <gunrock/container/array.hxx>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void inspect_device_array(int* result) {
  gunrock::array<int, 4> values{};
  for (int index = 0; index < 4; ++index) {
    values[index] = index * 3;
  }
  result[0] = values[2];
  result[1] = static_cast<int>(values.size());
  result[2] = values.empty() ? 1 : 0;
}

TEST(containers, array_matches_fixed_size_contract) {
  gunrock::array<int, 4> values{};
  EXPECT_FALSE(values.empty());
  EXPECT_EQ(values.size(), 4);
  EXPECT_EQ(values.max_size(), 4);
  EXPECT_NE(values.data(), nullptr);

  for (int index = 0; index < 4; ++index) {
    values[index] = index + 10;
  }
  gunrock::array<int, 4> const& const_values = values;
  EXPECT_EQ(const_values.data(), values.data());
  expect_values(const_values, {10, 11, 12, 13});

  gunrock::array<int, 0> empty{};
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.data(), nullptr);
}

TEST(containers, array_works_in_device_code) {
  thrust::device_vector<int> result(3, -1);
  inspect_device_array<<<1, 1>>>(result.data().get());
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  thrust::host_vector<int> host_result(result);
  expect_values(host_result, {6, 4, 0});
}