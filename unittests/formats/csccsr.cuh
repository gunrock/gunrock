#include <gunrock/algorithms/algorithms.hxx>

using namespace gunrock;
using namespace memory;

void test_csc_csr(int num_arguments, char** argument_array) {
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
  
  gunrock::io::loader<vertex_t,edge_t,weight_t> load_obj;
  load_obj = mm.load(filename);
   
  csr.from_coo(load_obj.coo);
  // Convert from CSR to CSC
  csc.from_csr(csr);

  // --
  // Build graph

  thrust::host_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::host_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  // Use both CSR and CSC views
  auto G = graph::build::build<memory_space_t::host>(load_obj.properties, csr, csc);
  
  std::cout << "Directed: " << G.is_directed() << "\n";
  std::cout << "Symmetric: " << G.is_symmetric() << "\n";
  std::cout << "Weighted: " << G.is_weighted() << "\n";

  // // Test CSR and CSC views
  // using csr_v_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  // using csc_v_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
  
  // // CSR number of vertices
  // std::cout << "G.get_number_of_vertices() : "
  //           << G.template get_number_of_vertices<csr_v_t>() << std::endl;
  // // CSC number of vertices
  // std::cout << "G.get_number_of_vertices() : "
  //           << G.template get_number_of_vertices<csc_v_t>() << std::endl;
  // // CSR number of edges
  // std::cout << "G.get_number_of_edges()    : "
  //           << G.template get_number_of_edges<csr_v_t>() << std::endl;
  // // CSC number of edges
  // std::cout << "G.get_number_of_edges()    : "
  //           << G.template get_number_of_edges<csc_v_t>() << std::endl;

  // for (vertex_t i = 0; i < G.template get_number_of_edges<csr_v_t>(); i++) {
  //   // Print CSR edge i
  //   std::cout << i << " " << G.template get_source_vertex<csr_v_t>(i) << " "
  //             << G.template get_destination_vertex<csr_v_t>(i) << std::endl;
  //   // Print CSC edge i
  //   std::cout << i << " " << G.template get_source_vertex<csc_v_t>(i) << " "
  //             << G.template get_destination_vertex<csc_v_t>(i) << std::endl;
  // }

  // // Print CSR edge index for edge from 6 -> 0
  // std::cout << G.template get_edge<csr_v_t>(6, 0) << std::endl;
  // // Print CSC edge index for edge from 6 -> 0
  // std::cout << G.template get_edge<csc_v_t>(6, 0) << std::endl;
}

int main(int argc, char** argv) {
  test_csc_csr(argc, argv);
}
