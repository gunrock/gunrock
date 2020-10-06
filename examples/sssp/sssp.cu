#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/error.hxx>            // error checking
#include <gunrock/graph/graph.hxx>      // graph class
#include <gunrock/formats/formats.hxx>  // csr support

#include <gunrock/applications/sssp/sssp.hxx>

using namespace gunrock;

void test_sssp() {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
}

int main(int argc, char** argv) {
  test_sssp();
  return EXIT_SUCCESS;
}