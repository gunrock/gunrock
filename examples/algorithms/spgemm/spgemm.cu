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

  // Filename to be read
  std::string filename_a = argument_array[1];
  constexpr memory_space_t space = memory_space_t::device;

  /// Load the matrix-market dataset into csr format.
  /// See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  using csr_t = format::csr_t<space, vertex_t, edge_t, weight_t>;
  csr_t a_csr;
  a_csr.from_coo(mm.load(filename_a));

  auto A = graph::build::from_csr<space, graph::view_t::csr>(
      a_csr.number_of_rows, a_csr.number_of_columns, a_csr.number_of_nonzeros,
      a_csr.row_offsets.data().get(), a_csr.column_indices.data().get(),
      a_csr.nonzero_values.data().get());

  std::string filename_b = argument_array[2];
  csr_t b_csr;
  b_csr.from_coo(mm.load(filename_b));

  /// For now, we are using the transpose of CSR-matrix A as the second operand
  /// for our spgemm.
  auto B = graph::build::from_csr<space, graph::view_t::csr>(
      b_csr.number_of_rows, b_csr.number_of_columns, b_csr.number_of_nonzeros,
      b_csr.row_offsets.data().get(), b_csr.column_indices.data().get(),
      b_csr.nonzero_values.data().get());

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