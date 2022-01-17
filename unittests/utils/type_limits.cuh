#include <cstdlib>
#include <iostream>

#include <gunrock/util/type_limits.hxx>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

void test_type_limits() {
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

  auto apply = [=] __device__(int const& x) {
    auto y = gunrock::numeric_limits<float>::invalid();
    bool v = gunrock::util::limits::is_valid(x);
    bool inv = gunrock::util::limits::is_valid(y);

    printf("%f\n", y);
  };

  thrust::for_each(thrust::device,
                   thrust::make_counting_iterator<int>(0),  // Begin: 0
                   thrust::make_counting_iterator<int>(1),  // End: 1
                   apply                                    // Unary Operator
  );
}

int main(int argc, char** argv) {
  test_type_limits();
  return EXIT_SUCCESS;
}