#include <cstdlib>
#include <iostream>

#include <gunrock/util/type_limits.hxx>

void test_type_limits() {
  float i = gunrock::numeric_limits<float>::invalid();
  std::cout << i << " is valid? " << std::boolalpha
            << gunrock::util::limits::is_valid(i) << std::endl;
}

int main(int argc, char** argv) {
  test_type_limits();
  return EXIT_SUCCESS;
}