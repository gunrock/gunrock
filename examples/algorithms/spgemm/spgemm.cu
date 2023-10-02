#include <gunrock/algorithms/spgemm.hxx>

using namespace gunrock;
using namespace memory;

void test_spmv(int num_arguments, char** argument_array) {
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/<program-name> a.mtx b.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types
  // Specify the types that will be used for
  // - vertex ids (vertex_t)
  // - edge offsets (edge_t)
  // - edge weights (weight_t)

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  constexpr memory_space_t space = memory_space_t::device;
  using csr_t = format::csr_t<space, vertex_t, edge_t, weight_t>;
  using csc_t = format::csc_t<space, vertex_t, edge_t, weight_t>;

  // Load A
  // Filename to be read
  std::string filename_a = argument_array[1];

  /// Load the matrix-market dataset into csr format.
  /// See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  csr_t a_csr;

  auto [a_properties, a_coo] = mm.load(filename_a);
  a_csr.from_coo(a_coo);

  // --
  // Build graph for A
  auto A = graph::build<memory_space_t::device>(a_properties, a_csr);

  // Load B
  // Filename to be read
  std::string filename_b = argument_array[2];

  /// Load the matrix-market dataset into csr format.
  /// See `format` to see other supported formats.
  csr_t b_csr;
  csc_t b_csc;

  auto [b_properties, b_coo] = mm.load(filename_b);

  b_csr.from_coo(b_coo);
  b_csc.from_csr(b_csr);

  // --
  // Build graph for B
  /// For now, we are using the transpose of CSR-matrix A as the second operand
  /// for our spgemm.
  auto B = graph::build<memory_space_t::device>(b_properties, b_csc, b_csr);

  /// Let's use CSR representation
  csr_t C;

  // --
  // GPU Run
  float gpu_elapsed = gunrock::spgemm::run(A, B, C);

  std::cout << "Number of rows: " << C.number_of_rows << std::endl;
  std::cout << "Number of columns: " << C.number_of_columns << std::endl;
  std::cout << "Number of nonzeros: " << C.number_of_nonzeros << std::endl;

  print::head(C.row_offsets, 10, "row_offsets");
  print::head(C.column_indices, 10, "column_indices");
  print::head(C.nonzero_values, 10, "nonzero_values");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}