#include <gunrock/algorithms/algorithms.hxx>

using namespace gunrock;
using namespace memory;

void test_coo(int num_arguments, char** argument_array) {
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
  using coo_t = format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  coo_t coo;
  csr.from_coo(mm.load(filename));

  coo.row_indices = thrust::host_vector<vertex_t>(csr.number_of_nonzeros);
  coo.column_indices = csr.column_indices;
  coo.nonzero_values = csr.nonzero_values;

  coo.number_of_columns = csr.number_of_columns;
  coo.number_of_rows = csr.number_of_rows;
  coo.number_of_nonzeros = csr.number_of_nonzeros;
  
  // --
  // Build graph
  auto G = graph::build::from_csr<memory_space_t::host, graph::view_t::coo>(
      csr, coo);

  // >>
  std::cout << "G.get_number_of_vertices()\t: " << G.get_number_of_vertices()
            << std::endl;
  std::cout << "G.get_number_of_edges()\t: " << G.get_number_of_edges()
            << std::endl;
  std::cout << "G.number_of_graph_representations()\t: "
            << G.number_of_graph_representations() << std::endl;

  gunrock::print::head(G.get_row_indices(), G.get_number_of_edges(),
                       G.get_number_of_edges());
  gunrock::print::head(G.get_column_indices(), G.get_number_of_edges(),
                       G.get_number_of_edges());
  gunrock::print::head(G.get_nonzero_values(), G.get_number_of_edges(),
                       G.get_number_of_edges());

  for (vertex_t i = 0; i < G.get_number_of_vertices(); i++)
    std::cout << i << " " << G.get_starting_edge(i) << std::endl;

  std::cout << "-------" << std::endl;

  for (vertex_t i = 0; i < G.get_number_of_vertices(); i++)
    std::cout << i << " " << G.get_number_of_neighbors(i) << std::endl;

  std::cout << "-------" << std::endl;

  std::cout << G.get_edge(0, 6) << std::endl;
  std::cout << G.get_edge(0, 7) << std::endl;
  std::cout << G.get_edge(38, 32) << std::endl;
}

int main(int argc, char** argv) {
  test_coo(argc, argv);
}
