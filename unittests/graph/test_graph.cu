#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/error.hxx>            // error checking
#include <gunrock/graph/graph.hxx>      // graph class
#include <gunrock/formats/formats.hxx>  // csr support

using namespace gunrock;

template <typename graph_ptr_type>
__global__ void kernel(graph_ptr_type G) {
  using vertex_t = typename graph_ptr_type::graph_type::vertex_type;
  using edge_t = typename graph_ptr_type::graph_type::edge_type;
  using weight_t = typename graph_ptr_type::graph_type::weight_type;

  using csr_view_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using csc_view_t = graph::graph_csc_t<vertex_t, edge_t, weight_t>;

  vertex_t source = 1;
  auto edge = 0;

  auto num_vertices = G->get_number_of_vertices();
  auto num_edges = G->get_number_of_edges();

  // Both valid.
  auto num_neighbors = G->template get_number_of_neighbors<csr_view_t>(source);
  auto source_vertex = G->csr_view_t::get_source_vertex(edge);
  auto edge_weight = G->get_edge_weight(edge);
  double average_degree = graph::get_average_degree(G);
  double degree_std_dev = graph::get_degree_standard_deviation(G);

  printf("__device__\n");
  printf("\tNumber of vertices: %i\n", num_vertices);
  printf("\tNumber of edges: %i\n", num_edges);
  printf("\tNumber of neighbors: %i (source = %i)\n", num_neighbors, source);
  printf("\tSource vertex: %i (edge = %i)\n", source_vertex, edge);
  printf("\tEdge weight: %f (edge = %i)\n", edge_weight, edge);
  printf("\tAverage degree: %lf\n", average_degree);
  printf("\tDegree std. deviation: %lf\n", degree_std_dev);
}

void test_graph() {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // Logical Matrix Representation
  // r/c  0 1 2 3
  //  0 [ 0 0 0 0 ]
  //  1 [ 5 8 0 0 ]
  //  2 [ 0 0 3 0 ]
  //  3 [ 0 6 0 0 ]

  // Logical Graph Representation
  // (i, j) [w]
  // (1, 0) [5]
  // (1, 1) [8]
  // (2, 2) [3]
  // (3, 1) [6]

  // CSR Matrix Representation
  // V            = [ 5 8 3 6 ]
  // COL_INDEX    = [ 0 1 2 1 ]
  // ROW_OFFSETS  = [ 0 0 2 3 4 ]
  vertex_t r = 4, c = 4;
  edge_t nnz = 4;

  // let's use thrust vector<type_t> for initial arrays
  thrust::host_vector<edge_t> h_Ap(r + 1);
  thrust::host_vector<vertex_t> h_Aj(nnz);
  thrust::host_vector<weight_t> h_Ax(nnz);

  auto Ap = h_Ap.data();
  auto Aj = h_Aj.data();
  auto Ax = h_Ax.data();

  Ap[0] = 0;
  Ap[1] = 0;
  Ap[2] = 2;
  Ap[3] = 3;
  Ap[4] = 4;
  Aj[0] = 0;
  Aj[1] = 1;
  Aj[2] = 2;
  Aj[3] = 3;
  Ax[0] = 5;
  Ax[1] = 8;
  Ax[2] = 3;
  Ax[3] = 6;

  using namespace memory;

  // wrap it with shared_ptr<csr_t> (memory_space_t::host)
  const graph::view_t graph_views = graph::view_t::csr;
  // graph::set(graph::view_t::csr, graph::view_t::csc);

  auto G = graph::build::from_csr<memory_space_t::host, graph_views>(
      r, c, nnz, h_Ap.data(), h_Aj.data(), h_Ax.data());

  using csr_view = graph::graph_csr_t<vertex_t, edge_t, weight_t>;

  vertex_t source = 1;
  vertex_t edge = 0;

  vertex_t num_vertices = G->get_number_of_vertices();
  edge_t num_edges = G->get_number_of_edges();
  edge_t num_neighbors = G->template get_number_of_neighbors<csr_view>(source);
  vertex_t source_vertex = G->template get_source_vertex<csr_view>(edge);
  weight_t edge_weight = G->template get_edge_weight<csr_view>(edge);
  double average_degree = graph::get_average_degree(G);
  double degree_std_dev = graph::get_degree_standard_deviation(G);

  // Host Output
  std::cout << "__host__" << std::endl;
  std::cout << "\tNumber of vertices: " << num_vertices << std::endl;
  std::cout << "\tNumber of edges: " << num_edges << std::endl;
  std::cout << "\tNumber of neighbors: " << num_neighbors
            << " (source = " << source << ")" << std::endl;
  std::cout << "\tSource vertex: " << source_vertex << " (edge = " << edge
            << ")" << std::endl;
  std::cout << "\tEdge weight: " << edge_weight << " (edge = " << edge << ")"
            << std::endl;
  std::cout << "\tAverage Degree: " << average_degree << std::endl;
  std::cout << "\tDegree Std. Deviation: " << degree_std_dev << std::endl;

  // Compile-Time Constants (device && host)
  std::cout << "\tNumber of Graph Representations = "
            << G->number_of_graph_representations() << std::endl;
  std::cout << "\tContains CSR Representation? " << std::boolalpha
            << G->template contains_representation<csr_view>() << std::endl;

  // wrap it with shared_ptr<csr_t> (memory_space_t::device)
  thrust::device_vector<edge_t> d_Ap = h_Ap;
  thrust::device_vector<vertex_t> d_Aj = h_Aj;
  thrust::device_vector<weight_t> d_Ax = h_Ax;

  auto O = graph::build::from_csr<memory_space_t::device, graph_views>(
      r, c, nnz, d_Ap.data().get(), d_Aj.data().get(), d_Ax.data().get());

  // Device Output
  cudaDeviceSynchronize();
  kernel<<<1, 1>>>(O);
  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());
}

int main(int argc, char** argv) {
  test_graph();
  return EXIT_SUCCESS;
}