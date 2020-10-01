// XXX: dummy template for unit testing

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <gunrock/error.hxx>
#include <gunrock/graph/graph.hxx>
#include <gunrock/formats/formats.hxx>

void test_graph()
{
  using namespace gunrock;

  using vertex_t  = int;
  using edge_t    = int;
  using weight_t  = float;

  error::error_t status = cudaSuccess;

  // XXX: (hide behind load) CSR array with space allocated (4x4x4)
  constexpr memory::memory_space_t space = memory::memory_space_t::host;

  using g_csr_t   = graph::graph_csr_t<vertex_t, edge_t, weight_t, space>;
  using g_csc_t   = graph::graph_csc_t<vertex_t, edge_t, weight_t, space>;
  using g_coo_t   = graph::graph_coo_t<vertex_t, edge_t, weight_t, space>;

  using csr_type = format::csr_t<edge_t, vertex_t, weight_t, space>;

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

  std::shared_ptr<csr_type> csr_ptr(
    new csr_type{ r, c, nnz, _Ap, _Aj, _Ax });

  // thrust::device_vector<edge_t> row_offsets       = _Ap;
  // thrust::device_vector<vertex_t> column_indices  = _Aj;
  // thrust::device_vector<weight_t> nonzero_values  = _Ax;

  // // wrap it with shared_ptr<csr_t> (memory_space_t::device)
  // std::shared_ptr<csr_type> csr_ptr(
  //   new csr_type{ r, c, nnz, row_offsets, column_indices, nonzero_values });

  graph::graph_t<vertex_t, edge_t, weight_t, space, g_csr_t, g_csc_t /* , g_coo_t*/> graph_slice;

  graph_slice.from_csr_t<csr_type>(csr_ptr);
  std::cout << "Number of Graph Representations = " 
            << graph_slice.number_of_graph_representations() << std::endl;
  std::cout << "Contains CSR Representation? " << std::boolalpha
            << graph_slice.contains_representation<g_csr_t>() << std::endl;

  vertex_t source = 1;
  vertex_t edge = 3;

  vertex_t num_vertices   = graph_slice.get_number_of_vertices();
  edge_t num_edges        = graph_slice.get_number_of_edges();
  edge_t num_neighbors    = graph_slice.get_neighbor_list_length(source);
  vertex_t source_vertex  = graph_slice.get_source_vertex(edge);  // XXX: doesn't work
  double average_degree   = graph::get_average_degree(graph_slice);
  double degree_std_dev   = graph::get_degree_standard_deviation(graph_slice);

  
  std::cout << "Average Degree: "       << average_degree << std::endl;
  std::cout << "Degree Std. Deviation: "<< degree_std_dev << std::endl;
  std::cout << "Number of vertices: "   << num_vertices   << std::endl;
  std::cout << "Number of edges: "      << num_edges      << std::endl;
  std::cout << "Number of neighbors: "  << num_neighbors 
            << " (source = "            << source << ")"  << std::endl;
  std::cout << "Source vertex: "        << source_vertex 
            << " (edge = "              << edge   << ")"  << std::endl;
}

int
main(int argc, char** argv)
{
  test_graph();
  return;
}