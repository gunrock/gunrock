#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/error.hxx>            // error checking
#include <gunrock/graph/graph.hxx>      // graph class
#include <gunrock/formats/formats.hxx>  // csr support

using namespace gunrock;
using namespace memory;

template <typename graph_type>
__host__ __device__ void use_graph(graph_type G) {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  using csr_view_t = typename graph_type::graph_csr_view_t;
  using csc_view_t = typename graph_type::graph_csc_view_t;
  using coo_view_t = typename graph_type::graph_coo_view_t;

  using use_type_t = csr_view_t;

  auto source = 2;
  auto edge = 1;

  auto num_vertices = G.get_number_of_vertices();
  auto num_edges = G.get_number_of_edges();

  // Both valid.
  auto num_neighbors = G.template get_number_of_neighbors<use_type_t>(source);
  auto source_vertex = G.template get_source_vertex<use_type_t>(edge);
  auto destination_vertex = G.template get_destination_vertex<use_type_t>(edge);
  auto edge_weight = G.template get_edge_weight<use_type_t>(edge);
  auto starting_edge = G.template get_starting_edge<use_type_t>(source);
  double average_degree = graph::get_average_degree(G);
  double degree_std_dev = graph::get_degree_standard_deviation(G);

  if constexpr (G.memory_space() == memory_space_t::host)
    printf("__host__\n");
  else
    printf("__device__\n");

  printf("\tNumber of vertices: %i\n", num_vertices);
  printf("\tNumber of edges: %i\n", num_edges);
  printf("\tNumber of neighbors: %i (source = %i)\n", num_neighbors, source);
  printf("\tSource vertex: %i (edge = %i)\n", source_vertex, edge);
  printf("\tDestination vertex: %i (edge = %i)\n", destination_vertex, edge);
  printf("\tEdge weight: %f (edge = %i)\n", edge_weight, edge);
  printf("\tStarting Edge: %i (vertex = %i)\n", starting_edge, source);
  printf("\tAverage degree: %lf\n", average_degree);
  printf("\tDegree std. deviation: %lf\n", degree_std_dev);
}

template <typename graph_type>
__global__ void kernel(graph_type G) {
  use_graph(G);
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
  thrust::host_vector<vertex_t> h_J(nnz);
  thrust::host_vector<weight_t> h_Ax(nnz);
  thrust::host_vector<vertex_t> h_I(nnz);
  thrust::host_vector<edge_t> h_Aj(c + 1);

  auto Ap = h_Ap.data();
  auto J = h_J.data();
  auto Ax = h_Ax.data();

  Ap[0] = 0;
  Ap[1] = 0;
  Ap[2] = 2;
  Ap[3] = 3;
  Ap[4] = 4;
  J[0] = 0;
  J[1] = 1;
  J[2] = 2;
  J[3] = 3;
  Ax[0] = 5;
  Ax[1] = 8;
  Ax[2] = 3;
  Ax[3] = 6;

  // wrap it with shared_ptr<csr_t> (memory_space_t::host)
  const graph::view_t graph_views = /* graph::view_t::csr; */
      graph::set(graph::view_t::csr, graph::view_t::csc);

  auto G = graph::build::from_csr<memory_space_t::host, graph_views>(
      r, c, nnz, h_Ap.data(), h_J.data(), h_Ax.data(), h_I.data(), h_Aj.data());

  using csr_view_t = graph::graph_csr_t<vertex_t, edge_t, weight_t>;

  use_graph(G);

  // Compile-Time Constants (device && host)
  std::cout << "\tNumber of Graph Representations = "
            << G.number_of_graph_representations() << std::endl;
  std::cout << "\tContains CSR Representation? " << std::boolalpha
            << G.template contains_representation<csr_view_t>() << std::endl;

  // wrap it with shared_ptr<csr_t> (memory_space_t::device)
  thrust::device_vector<edge_t> d_Ap = h_Ap;
  thrust::device_vector<vertex_t> d_J = h_J;
  thrust::device_vector<weight_t> d_Ax = h_Ax;
  thrust::device_vector<vertex_t> d_I(nnz);
  thrust::device_vector<edge_t> d_Aj(c + 1);

  auto O = graph::build::from_csr<memory_space_t::device, graph_views>(
      r, c, nnz, d_Ap.data().get(), d_J.data().get(), d_Ax.data().get(),
      d_I.data().get(), d_Aj.data().get());

  // Device Output
  cudaDeviceSynchronize();
  kernel<<<1, 1>>>(O);
  cudaDeviceSynchronize();
  error::throw_if_exception(cudaPeekAtLastError());

  // TODO: Revisit this test.
  // thrust::device_vector<vertex_t> histogram(sizeof(vertex_t) * 8 + 1);
  // gunrock::graph::build_degree_histogram(G, histogram.data().get());

  // std::cout << "Degree Histogram = ";
  // thrust::copy(histogram.begin(), histogram.end(),
  //              std::ostream_iterator<vertex_t>(std::cout, " "));
  // std::cout << std::endl;
}

int main(int argc, char** argv) {
  test_graph();
  return EXIT_SUCCESS;
}