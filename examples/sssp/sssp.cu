#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/sssp/sssp.hxx>

using namespace gunrock;
using namespace memory;

void test_sssp(int num_arguments, char** argument_array) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/color filename.mtx" << std::endl;
    exit(1);
  }

  // Load Matrix-Market file & convert the resultant COO into CSR format.
  std::string filename = argument_array[1];
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto coo = mm.load(filename);

  // convert coo to csr
  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr = coo;  // Able to convert host-based coo_t to device-based csr (or host
              // to host). As of right now, it requires coo to be host side.

  vertex_t source = 0;
  thrust::device_vector<weight_t> d_distances(csr.number_of_rows);

  // calling sssp
  float elapsed = sssp::execute(csr,         // device csr_t sparse data
                                source,      // single source
                                d_distances  // output distances
  );

  std::cout << "Distances (output) = ";
  thrust::copy(d_distances.begin(), d_distances.end(),
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "SSSP Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
  return EXIT_SUCCESS;
}
