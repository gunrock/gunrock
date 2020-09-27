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

  constexpr bool HAS_CSR = false;
  constexpr bool HAS_CSC = false;
  constexpr bool HAS_COO = false;

  error::error_t status = cudaSuccess;

  // graph::graph_t<HAS_COO, HAS_CSR, HAS_CSC,
  //         vertex_t, edge_t, weight_t> graph;

  // XXX: (hide behind load) CSR array with space allocated (4x4x4)
  std::size_t r = 4, c = 4, nnz = 4;
  memory::memory_space_t location = memory::memory_space_t::host;

  // CSR Matrix Representation
  // V            = [ 5 8 3 6 ]
  // COL_INDEX    = [ 0 1 2 1 ]
  // ROW_OFFSETS  = [ 0 0 2 3 4 ]
  edge_t *Ap = memory::allocate<edge_t>((r+1)*sizeof(edge_t), location);
  vertex_t *Aj = memory::allocate<vertex_t>((nnz)*sizeof(vertex_t), location);
  weight_t *Ax = memory::allocate<weight_t>((nnz)*sizeof(weight_t), location);

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

  // XXX: ugly way to initialize these, but it works.
  Ap[0] = 0; Ap[1] = 0; Ap[2] = 2; Ap[3] = 3; Ap[4] = 4;
  Aj[0] = 0; Aj[1] = 1; Aj[2] = 2; Aj[3] = 3;
  Ax[0] = 5; Ax[1] = 8; Ax[2] = 3; Ax[3] = 6;

  format::csr_t<edge_t, vertex_t, weight_t> csr(r, c, nnz, Ap, Aj, Ax, location);

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