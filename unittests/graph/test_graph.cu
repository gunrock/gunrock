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

  using g_csr_t   = graph::graph_csr_t<vertex_t, edge_t, weight_t>;
  using g_csc_t   = graph::graph_csc_t<vertex_t, edge_t, weight_t>;
  using g_coo_t   = graph::graph_coo_t<vertex_t, edge_t, weight_t>;

  error::error_t status = cudaSuccess;

  // XXX: (hide behind load) CSR array with space allocated (4x4x4)
  memory::memory_space_t location = memory::memory_space_t::host;

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

  using csr_t = format::csr_t<edge_t, vertex_t, weight_t>;

  // XXX: ugly way to initialize these, but it works.
  csr_t csr;

  csr.num_rows = csr.num_columns = csr.num_nonzeros = 4;

  csr.row_offsets = std::shared_ptr<edge_t>(
                      memory::allocate<edge_t>(
                        (csr.num_rows+1)*sizeof(edge_t), location),
                      [&](edge_t* ptr){ memory::free(ptr, location); });

  csr.column_indices = std::shared_ptr<vertex_t>(
                        memory::allocate<vertex_t>(
                          (csr.num_nonzeros)*sizeof(vertex_t), location),
                        [&](vertex_t* ptr){ memory::free(ptr, location); });

  csr.nonzero_values = std::shared_ptr<weight_t>(
                        memory::allocate<weight_t>(
                          (csr.num_nonzeros)*sizeof(weight_t), location),
                        [&](weight_t* ptr){ memory::free(ptr, location); });
  
  auto Ap = csr.row_offsets.get();
  auto Aj = csr.column_indices.get();
  auto Ax = csr.nonzero_values.get();

  Ap[0] = 0; Ap[1] = 0; Ap[2] = 2; Ap[3] = 3; Ap[4] = 4;
  Aj[0] = 0; Aj[1] = 1; Aj[2] = 2; Aj[3] = 3;
  Ax[0] = 5; Ax[1] = 8; Ax[2] = 3; Ax[3] = 6;

  graph::graph_t<vertex_t, edge_t, weight_t, g_csr_t, g_csc_t /* , g_coo_t*/> graph_slice;

  graph_slice.from_csr_t(csr);
  std::cout << "Number of Graph Representations = " 
            << graph_slice.number_of_graph_representations() << std::endl;
  std::cout << "Contains CSR Representation? " << std::boolalpha
            << graph_slice.contains_representation<g_csr_t>() << std::endl;

  vertex_t source = 1;
  vertex_t edge = 1;

  vertex_t num_vertices   = graph_slice.get_number_of_vertices();
  edge_t num_edges        = graph_slice.get_number_of_edges();
  edge_t num_neighbors    = graph_slice.get_neighbor_list_length(source);
  vertex_t source_vertex  = graph_slice.get_source_vertex(edge);
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