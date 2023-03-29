#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/error.hxx>            // error checking
#include <gunrock/formats/formats.hxx>  // csr support

void test_format() {
  using namespace gunrock;
  using namespace gunrock::format;

  using offset_t = int;
  using index_t = int;
  using value_t = float;

  error::error_t status = cudaSuccess;

  // CSR, CSC, COO classes with default constructors
  csr_t<index_t, offset_t, value_t> csr;
  csc_t<index_t, offset_t, value_t> csc;
  coo_t<index_t, index_t, value_t> coo;
}

int main(int argc, char** argv) {
  test_format();
  return EXIT_SUCCESS;
}