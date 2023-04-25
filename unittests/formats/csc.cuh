#include <gunrock/algorithms/algorithms.hxx>

void test_csc(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  std::string filename = argument_array[1];

  using namespace gunrock;
  using namespace memory;

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  // Test CSC
  format::csc_t<memory_space_t::host, vertex_t, edge_t, weight_t> csc;
  format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(coo);
  csc.from_csr(csr);

  // CSC
  std::cout << "Row Indices: ";
  for (auto index : csc.row_indices)
    std::cout << index << " ";
  std::cout << std::endl;

  std::cout << "Column Offsets: ";
  for (auto offset : csc.column_offsets)
    std::cout << offset << " ";
  std::cout << std::endl;

  std::cout << "Nonzero Values: ";
  for (auto nz : csc.nonzero_values)
    std::cout << nz << " ";
  std::cout << std::endl;

  // Use CSC view
  auto G = graph::build<memory_space_t::host>(properties, csc);

  // Test graph properties
  std::cout << "Directed: " << G.is_directed() << "\n";
  std::cout << "Symmetric: " << G.is_symmetric() << "\n";
  std::cout << "Weighted: " << G.is_weighted() << "\n";

  // Test CSC view
  using csr_v_t =
      graph::graph_csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  using csc_v_t =
      graph::graph_csc_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  using coo_v_t =
      graph::graph_coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;

  // CSC number of vertices
  std::cout << "G.get_number_of_vertices<csc_v_t>() : "
            << G.template get_number_of_vertices<csc_v_t>() << std::endl;
  // CSC number of edges
  std::cout << "G.get_number_of_edges<csc_v_t>()    : "
            << G.template get_number_of_edges<csc_v_t>() << std::endl;

  for (vertex_t i = 0; i < G.template get_number_of_edges<csc_v_t>(); i++) {
    // Print CSC edge i
    std::cout << i << " " << G.template get_source_vertex<csc_v_t>(i) << " "
              << G.template get_destination_vertex<csc_v_t>(i) << std::endl;
  }

  // Print CSR edge index for edge from 6 -> 0
  std::cout << G.template get_edge<csc_v_t>(6, 0) << std::endl;
}

int main(int argc, char** argv) {
  test_csc(argc, argv);
}