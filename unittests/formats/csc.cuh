#include <gunrock/algorithms/algorithms.hxx>

using namespace gunrock;
using namespace memory;

void test_csc(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  using csr_t = format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  csr_t csr;
  using csc_t = format::csc_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  csc_t csc;
  csr.from_coo(mm.load(filename));
  csc.from_csr(csr);

  // --
  // Build graph

  thrust::host_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::host_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  auto G =
      graph::build::build<memory_space_t::host>(
          csc);  // supports row_indices and column_offsets (default = nullptr)

  // >>
  std::cout << "G.get_number_of_vertices() : " << G.get_number_of_vertices()
            << std::endl;
  std::cout << "G.get_number_of_edges()    : " << G.get_number_of_edges()
            << std::endl;

  // gunrock::print::head(G.get_column_offsets(),
  // G.get_number_of_vertices(), G.get_number_of_vertices());
  // gunrock::print::head(G.get_row_indices(),
  // G.get_number_of_edges(), G.get_number_of_edges());
  // gunrock::print::head(G.get_nonzero_values(),
  // G.get_number_of_edges(), G.get_number_of_edges());

  // for(vertex_t i = 0; i < G.get_number_of_edges(); i++)
  //   std::cout << i << " " << G.get_source_vertex(i) << " " <<
  //   G.get_destination_vertex(i) << std::endl;

  // std::cout << "-------" << std::endl;

  // std::cout << G.get_edge(6, 0) << std::endl;
  // std::cout << G.get_edge(0, 6) << std::endl;
  // std::cout << G.get_edge(38, 32) << std::endl;
  // std::cout << G.get_edge(32, 38) << std::endl;
}

int main(int argc, char** argv) {
  test_csc(argc, argv);
}
