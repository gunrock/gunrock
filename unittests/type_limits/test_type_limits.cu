#include <cstdlib>
#include <iostream>

#include <gunrock/util/type_limits.hxx>

void test_type_limits() {
  using type_t = short unsigned int;
  type_t i = gunrock::numeric_limits<type_t>::invalid();
  std::cout << "i = " << i << " (is valid? " << std::boolalpha
            << gunrock::util::limits::is_valid(i) << ")" << std::endl;
}

int main(int argc, char** argv) {
  test_type_limits();
  return EXIT_SUCCESS;
}