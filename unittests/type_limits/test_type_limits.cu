#include <cstdlib>
#include <iostream>

#include <gunrock/util/type_limits.hxx>

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
}

int main(int argc, char** argv) {
  test_type_limits();
  return EXIT_SUCCESS;
}