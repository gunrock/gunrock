#include <gunrock/algorithms/algorithms.hxx>

void test_coo_csr(int num_arguments, char** argument_array) {
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
  auto [properties, coo_load] = mm.load(filename);

  // Test COO and CSR
  format::coo_t<memory_space_t::host, vertex_t, edge_t, weight_t> coo;
  format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(coo_load);
  coo.from_csr(csr);

  // CSR
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

  // COO
  std::cout << "Row indices: ";
  for (auto i : coo.row_indices)
    std::cout << i << " ";
  std::cout << std::endl;

  std::cout << "Column Indices: ";
  for (auto j : coo.column_indices)
    std::cout << j << " ";
  std::cout << std::endl;

  std::cout << "Nonzero Values: ";
  for (auto nz : coo.nonzero_values)
    std::cout << nz << " ";
  std::cout << std::endl;

  // Use COO and CSR views
  auto G = graph::build<memory_space_t::host>(properties, coo, csr);

  // Test graph properties
  std::cout << "Directed: " << G.is_directed() << "\n";
  std::cout << "Symmetric: " << G.is_symmetric() << "\n";
  std::cout << "Weighted: " << G.is_weighted() << "\n";

  // Test COO and CSR views
  using csr_v_t =
      graph::graph_csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  using coo_v_t =
      graph::graph_coo_t<memory_space_t::host, vertex_t, edge_t, weight_t>;

  // CSR number of vertices
  std::cout << "G.get_number_of_vertices<csr_v_t>() : "
            << G.template get_number_of_vertices<csr_v_t>() << std::endl;
  // COO number of vertices
  std::cout << "G.get_number_of_vertices<coo_v_t>() : "
            << G.template get_number_of_vertices<coo_v_t>() << std::endl;
  // CSR number of edges
  std::cout << "G.get_number_of_edges<csr_v_t>()    : "
            << G.template get_number_of_edges<csr_v_t>() << std::endl;
  // COO number of edges
  std::cout << "G.get_number_of_edges<coo_v_t>()    : "
            << G.template get_number_of_edges<coo_v_t>() << std::endl;

  for (vertex_t i = 0; i < G.template get_number_of_edges<csr_v_t>(); i++) {
    // Print CSR edge i
    std::cout << i << " " << G.template get_source_vertex<csr_v_t>(i) << " "
              << G.template get_destination_vertex<csr_v_t>(i) << std::endl;
    // Print COO edge i
    std::cout << i << " " << G.template get_source_vertex<coo_v_t>(i) << " "
              << G.template get_destination_vertex<coo_v_t>(i) << std::endl;
  }

  // Print CSR edge index for edge from 6 -> 0
  std::cout << G.template get_edge<csr_v_t>(6, 0) << std::endl;
  // Print COO edge index for edge from 6 -> 0
  std::cout << G.template get_edge<coo_v_t>(6, 0) << std::endl;
}

int main(int argc, char** argv) {
  test_coo_csr(argc, argv);
}