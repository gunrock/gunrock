#include <gunrock/error.hxx>             // error checking
#include <gunrock/formats/formats.hxx>   // formats (csr, coo)
#include <gunrock/graph/graph.hxx>       // graph class
#include <gunrock/memory.hxx>            // memory space
#include <gunrock/io/matrix_market.hxx>  // matrix_market support

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
void __global__ graph_op_kernel(graph_type* G) {
  printf("%i, %i, %i, %i\n", G->get_starting_edge(0),
         G->get_number_of_neighbors(0), G->get_number_of_vertices(),
         G->get_number_of_edges());
}

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
  format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t> csr;
  csr = coo;

  auto g = graph::build::from_csr_t<memory_space_t::host>(
      csr.number_of_rows,      // number of rows
      csr.number_of_columns,   // number of columns
      csr.number_of_nonzeros,  // number of edges
      csr.row_offsets, csr.column_indices, csr.nonzero_values);

  graph_op(g.data());

  thrust::device_vector<edge_t> d_Ap = csr.row_offsets;
  thrust::device_vector<vertex_t> d_Aj = csr.column_indices;
  thrust::device_vector<weight_t> d_Ax = csr.nonzero_values;

  auto G = graph::build::from_csr_t<memory_space_t::device>(
      csr.number_of_rows,      // number of rows
      csr.number_of_columns,   // number of columns
      csr.number_of_nonzeros,  // number of edges
      d_Ap, d_Aj, d_Ax);
  cudaDeviceSynchronize();
  graph_op_kernel<<<1, 1>>>(G.data().get());
  cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
  test_coo_to_csr(argc, argv);
}