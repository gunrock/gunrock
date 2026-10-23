#include <gunrock/error.hxx>             // error checking
#include <gunrock/io/matrix_market.hxx>  // matrix_market support

void test_matrix_market(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./test_matrix_market filename.mtx" << std::endl;
    exit(1);
  }

  std::string filename = argument_array[1];

  using namespace gunrock;

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
}

int main(int argc, char** argv) {
  test_matrix_market(argc, argv);
}