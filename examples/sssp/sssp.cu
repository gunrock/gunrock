#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/applications/sssp/sssp.hxx>

using namespace gunrock;

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
  format::csr_t<memory::memory_space_t::host, vertex_t, edge_t, weight_t> h_csr;
  h_csr = coo;

  // Move data to device.
  format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t>
      d_csr;

  d_csr.number_of_rows = h_csr.number_of_rows;
  d_csr.number_of_columns = h_csr.number_of_columns;
  d_csr.number_of_nonzeros = h_csr.number_of_nonzeros;
  d_csr.row_offsets = h_csr.row_offsets;
  d_csr.column_indices = h_csr.column_indices;
  d_csr.nonzero_values = h_csr.nonzero_values;

  vertex_t source = 0;
  thrust::device_vector<weight_t> d_distances(h_csr.number_of_rows);

  // calling sssp
  float elapsed = sssp::execute(h_csr, d_csr,
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
