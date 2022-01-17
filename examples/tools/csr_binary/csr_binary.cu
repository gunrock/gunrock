#include <gunrock/algorithms/algorithms.hxx>

using namespace gunrock;
using namespace memory;

void mtx2bin(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: " << argument_array[0] << " <inpath>" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  std::string inpath = argument_array[1];
  std::string outpath = inpath + ".csr";

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  using csr_t =
      format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.from_coo(mm.load(inpath));

  std::cout << "csr.number_of_rows     = " << csr.number_of_rows << std::endl;
  std::cout << "csr.number_of_columns  = " << csr.number_of_columns
            << std::endl;
  std::cout << "csr.number_of_nonzeros = " << csr.number_of_nonzeros
            << std::endl;
  std::cout << "writing to             = " << outpath << std::endl;

  csr.write_binary(outpath);
}

int main(int argc, char** argv) {
  mtx2bin(argc, argv);
}