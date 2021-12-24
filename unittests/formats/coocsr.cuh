#include <gunrock/error.hxx>             // error checking
#include <gunrock/formats/formats.hxx>   // formats (csr, coo)
#include <gunrock/memory.hxx>            // memory space
#include <gunrock/io/matrix_market.hxx>  // matrix_market support

void test_coo_to_csr(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./test_coo_to_csr filename.mtx" << std::endl;
    exit(1);
  }

  std::string filename = argument_array[1];

  using namespace gunrock;
  using namespace memory;

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto coo = mm.load(filename);
  for (int i = 0; i < coo.number_of_nonzeros; ++i) {
    std::cout << "(" << coo.row_indices[i] << ", " << coo.column_indices[i]
              << ")"
              << " = " << coo.nonzero_values[i] << std::endl;
  }

  format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> csr;
  csr = coo;

  std::cout << "Row offsets: ";
  for (auto offset : csr.row_offsets)
    std::cout << offset << " ";
  std::cout << std::endl;

  std::cout << "Column Indices: ";
  for (auto j : csr.column_indices)
    std::cout << j << " ";
  std::cout << std::endl;

  std::cout << "Nonzero Values: ";
  for (auto nz : csr.nonzero_values)
    std::cout << nz << " ";
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  test_coo_to_csr(argc, argv);
}