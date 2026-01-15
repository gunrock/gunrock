#include <iostream>

#include <gunrock/util/type_limits.hxx>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <gtest/gtest.h>

// Device function to test type limits - defined outside test method to avoid access issues
__device__ void test_type_limits_device_func(int const& x) {
  auto y = gunrock::numeric_limits<float>::invalid();
  bool v = gunrock::util::limits::is_valid(x);
  bool inv = gunrock::util::limits::is_valid(y);

  printf("%f\n", y);
  (void)v;
  (void)inv;
}

TEST(utils, type_limits) {
  std::cout << "invalid = " << gunrock::numeric_limits<int>::invalid()
            << " (is valid? " << std::boolalpha
            << gunrock::util::limits::is_valid(
                   gunrock::numeric_limits<int>::invalid())
            << ")" << std::endl;

  std::cout << "invalid = " << gunrock::numeric_limits<float>::invalid()
            << " (is valid? " << std::boolalpha
            << gunrock::util::limits::is_valid(
                   gunrock::numeric_limits<float>::invalid())
            << ")" << std::endl;

  std::cout << "invalid = " << gunrock::numeric_limits<double>::invalid()
            << " (is valid? " << std::boolalpha
            << gunrock::util::limits::is_valid(
                   gunrock::numeric_limits<double>::invalid())
            << ")" << std::endl;

  std::cout << "invalid = " << gunrock::numeric_limits<unsigned int>::invalid()
            << " (is valid? " << std::boolalpha
            << gunrock::util::limits::is_valid(
                   gunrock::numeric_limits<unsigned int>::invalid())
            << ")" << std::endl;

  // Use function pointer instead of lambda to avoid access issues with __device__ lambdas
  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator<int>(0),  // Begin: 0
                   thrust::make_counting_iterator<int>(1),  // End: 1
                   test_type_limits_device_func              // Function pointer instead of lambda
  );
}