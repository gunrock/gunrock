#include <gunrock/error.hxx>             // error checking
#include <gunrock/formats/formats.hxx>   // formats (csr, coo)
#include <gunrock/graph/graph.hxx>       // graph class
#include <gunrock/memory.hxx>            // memory space
#include <gunrock/io/matrix_market.hxx>  // matrix_market support
#include <gunrock/compat/runtime_api.h>

#include <gtest/gtest.h>

template <typename graph_type>
void graph_op(graph_type* g) {
  std::cout << g->get_starting_edge(0) << std::endl;
  std::cout << g->get_number_of_neighbors(0) << std::endl;

  std::cout << g->get_starting_edge(1) << std::endl;
  std::cout << g->get_number_of_neighbors(1) << std::endl;

  std::cout << g->get_starting_edge(2) << std::endl;
  std::cout << g->get_number_of_neighbors(2) << std::endl;

  std::cout << g->get_number_of_neighbors(6) << std::endl;

  std::cout << g->get_number_of_vertices() << std::endl;
  std::cout << g->get_number_of_edges() << std::endl;
}

template <typename graph_type>
__global__ void graph_op_kernel(graph_type G) {
  printf("%i, %i, %i, %i\n", G.get_starting_edge(0),
         G.get_number_of_neighbors(0), G.get_number_of_vertices(),
         G.get_number_of_edges());
}

TEST(graph, graph_load) {
  // Use a small sample graph for testing
  // This test creates a simple graph programmatically instead of loading from file
  using namespace gunrock;
  using namespace memory;

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // Create a simple graph manually (4 vertices, 4 edges)
  // This avoids needing a .mtx file
  format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> csr(4, 4, 4);
  csr.row_offsets[0] = 0;
  csr.row_offsets[1] = 0;
  csr.row_offsets[2] = 2;
  csr.row_offsets[3] = 3;
  csr.row_offsets[4] = 4;
  csr.column_indices[0] = 0;
  csr.column_indices[1] = 1;
  csr.column_indices[2] = 2;
  csr.column_indices[3] = 1;
  csr.nonzero_values[0] = 5;
  csr.nonzero_values[1] = 8;
  csr.nonzero_values[2] = 3;
  csr.nonzero_values[3] = 6;

  auto g = graph::build<memory_space_t::host>({}, csr);

  graph_op(&g);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> d_csr(csr);

  auto G = graph::build<memory_space_t::device>({}, d_csr);
  hipDeviceSynchronize();
  graph_op_kernel<<<1, 1>>>(G);
  hipDeviceSynchronize();
  error::throw_if_exception(hipGetLastError());
}