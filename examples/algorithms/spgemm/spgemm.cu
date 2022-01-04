#include <gunrock/algorithms/spgemm.hxx>

using namespace gunrock;
using namespace memory;

void test_spmv(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
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
  std::string filename = argument_array[1];
  constexpr memory_space_t space = memory_space_t::device;

  /// Load the matrix-market dataset into csr format.
  /// See `format` to see other supported formats.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  using csr_t = format::csr_t<space, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.from_coo(mm.load(filename));

  auto A = graph::build::from_csr<space, graph::view_t::csr>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<vertex_t> column_offsets(csr.number_of_columns + 1);

  /// For now, we are using the transpose of CSR-matrix A as the second operand
  /// for our spgemm.
  auto B = graph::build::from_csr<space, graph::view_t::csc>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get(), row_indices.data().get(),
      column_offsets.data().get());

  /// We will use the following graph in csr view to store the sparse-matrix C's
  /// result. Initially, we only know the m x n matrix size of C, which is the
  /// number of rows of A (m) and the number of columns of B (n). The number of
  /// nonzeros of C is unknown (and is therefore set to 0). C must be in the CSR
  /// format for essentials.
  csr_t cc(csr.number_of_rows, csr.number_of_columns, 0);
  using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  graph::graph_t<space, vertex_t, edge_t, weight_t, csr_v_t> C;
  C.template set<csr_v_t>(
      cc.number_of_rows, cc.number_of_nonzeros, cc.row_offsets.data().get(),
      cc.column_indices.data().get(), cc.nonzero_values.data().get());

  // --
  // GPU Run
  float gpu_elapsed = gunrock::spgemm::run(A, B, C);
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}