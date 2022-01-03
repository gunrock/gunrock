#include <gunrock/algorithms/hits.hxx>

using namespace gunrock;
using namespace memory;

void test_hits(int argc, char** argv) {
  if (2 != argc) {
    std::cerr << "usage:: ./bin/<program-name> filename.mtx \n";
    exit(1);
  }

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  csr_t csr;

  std::string filename = argv[1];
  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );

  hits::param_c param{20};
  hits::result_c result{G};

  auto time = gunrock::hits::run(G, param, result);
  result.print_result(20);
}

int main(int argc, char** argv) {
  test_hits(argc, argv);
  return 0;
}
