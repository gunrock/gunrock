#include <cstdlib>                      // EXIT_SUCCESS

#include <gunrock/error.hxx>            // error checking
#include <gunrock/graph/graph.hxx>      // graph class
#include <gunrock/formats/formats.hxx>  // csr support

using namespace gunrock;

template<typename graph_type>
__global__ void kernel(graph_type* graph) {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  vertex_t source = 1;
  vertex_t edge = 0;

  vertex_t num_vertices   = graph->get_number_of_vertices();
  edge_t num_edges        = graph->get_number_of_edges();
  edge_t num_neighbors    = graph->get_neighbor_list_length(source);
  vertex_t source_vertex  = graph->get_source_vertex(edge);
  // double average_degree   = graph::get_average_degree(graph);
  // double degree_std_dev   = graph::get_degree_standard_deviation(graph);

  printf("__device__\n");
  // printf("\tAverage degree: %lf", average_degree);
  // printf("\tDegree std. deviation: %lf", degree_std_dev);
  printf("\tNumber of vertices: %i\n", num_vertices);
  printf("\tNumber of edges: %i\n", num_edges);
  printf("\tNumber of neighbors: %i (source = %i)\n", num_neighbors, source);
  printf("\tSource vertex: %i (edge = %i)\n", source_vertex, edge);
}

void test_graph()
{
  using vertex_t  = int;
  using edge_t    = int;
  using weight_t  = float;

  // error::error_t status = cudaSuccess;

  // XXX: (hide behind load) CSR array with space allocated (4x4x4)
  constexpr memory::memory_space_t space = memory::memory_space_t::device;

  using g_csr_t   = graph::graph_csr_t<vertex_t, edge_t, weight_t, space>;
  using g_csc_t   = graph::graph_csc_t<vertex_t, edge_t, weight_t, space>;
  using g_coo_t   = graph::graph_coo_t<vertex_t, edge_t, weight_t, space>;

  using csr_type = format::csr_t<edge_t, vertex_t, weight_t, space>;

  constexpr memory::memory_space_t host_space = memory::memory_space_t::host;

  using host_g_csr_t   = graph::graph_csr_t<vertex_t, edge_t, weight_t, host_space>;
  using host_g_csc_t   = graph::graph_csc_t<vertex_t, edge_t, weight_t, host_space>;
  using host_g_coo_t   = graph::graph_coo_t<vertex_t, edge_t, weight_t, host_space>;

  using host_csr_type = format::csr_t<edge_t, vertex_t, weight_t, host_space>;

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
  vertex_t r = 4, c = 4, nnz = 4;

  // let's use thrust vector<type_t> for initial arrays
  thrust::host_vector<edge_t>   _Ap(r+1);
  thrust::host_vector<vertex_t> _Aj(nnz);
  thrust::host_vector<weight_t> _Ax(nnz);

  auto Ap = _Ap.data();
  auto Aj = _Aj.data();
  auto Ax = _Ax.data();

  Ap[0] = 0; Ap[1] = 0; Ap[2] = 2; Ap[3] = 3; Ap[4] = 4;
  Aj[0] = 0; Aj[1] = 1; Aj[2] = 2; Aj[3] = 3;
  Ax[0] = 5; Ax[1] = 8; Ax[2] = 3; Ax[3] = 6;

  // wrap it with shared_ptr<csr_t> (memory_space_t::host)
  std::shared_ptr<host_csr_type> host_csr_ptr(
    new host_csr_type{ r, c, nnz, _Ap, _Aj, _Ax });

  graph::graph_t<vertex_t, edge_t, weight_t, 
    host_space, host_g_csr_t, host_g_csc_t /* , host_g_coo_t*/> host_graph_slice;

  // Intialize graph using a CSR-type
  host_graph_slice.from_csr_t<host_csr_type>(host_csr_ptr);

  vertex_t source = 1;
  vertex_t edge = 0;

  vertex_t num_vertices   = host_graph_slice.get_number_of_vertices();
  edge_t num_edges        = host_graph_slice.get_number_of_edges();
  edge_t num_neighbors    = host_graph_slice.get_neighbor_list_length(source);
  vertex_t source_vertex  = host_graph_slice.get_source_vertex(edge);
  double average_degree   = graph::get_average_degree(host_graph_slice);
  double degree_std_dev   = graph::get_degree_standard_deviation(host_graph_slice);

  // Host Output
  std::cout << "Average Degree: "       << average_degree << std::endl;
  std::cout << "Degree Std. Deviation: "<< degree_std_dev << std::endl;
  std::cout << "Number of vertices: "   << num_vertices   << std::endl;
  std::cout << "Number of edges: "      << num_edges      << std::endl;
  std::cout << "Number of neighbors: "  << num_neighbors 
            << " (source = "            << source << ")"  << std::endl;
  std::cout << "Source vertex: "        << source_vertex 
            << " (edge = "              << edge   << ")"  << std::endl;

  // Compile-Time Constants (device && host)
  std::cout << "Number of Graph Representations = " 
            << host_graph_slice.number_of_graph_representations() << std::endl;
  std::cout << "Contains CSR Representation? " << std::boolalpha
            << host_graph_slice.contains_representation<g_csr_t>() << std::endl;

  // wrap it with shared_ptr<csr_t> (memory_space_t::device)
  thrust::device_vector<edge_t> row_offsets       = _Ap;
  thrust::device_vector<vertex_t> column_indices  = _Aj;
  thrust::device_vector<weight_t> nonzero_values  = _Ax;

  std::shared_ptr<csr_type> csr_ptr(
    new csr_type{ r, c, nnz, row_offsets, column_indices, nonzero_values });

  using graph_type = graph::graph_t<vertex_t, edge_t, weight_t, space, g_csr_t, g_csc_t /* , g_coo_t*/>;

  std::shared_ptr<graph_type> graph_slice_ptr;
  auto safe_raw_ptr = graph_slice_ptr.get();
  
  // Intialize graph using a CSR-type
  safe_raw_ptr->from_csr_t<csr_type>(csr_ptr);

  // Device Output
  cudaDeviceSynchronize();
  kernel<<<1, 1>>>(safe_raw_ptr);
  cudaDeviceSynchronize();
}

int
main(int argc, char** argv)
{
  test_graph();
  return EXIT_SUCCESS;
}