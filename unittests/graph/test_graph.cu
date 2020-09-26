// XXX: dummy template for unit testing

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <gunrock/error.hxx>
#include <gunrock/graph/graph.hxx>

void test_graph()
{
  using namespace gunrock;

  using vertex_t  = int;
  using edge_t    = int;
  using weight_t  = float;

  constexpr bool HAS_CSR = false;
  constexpr bool HAS_CSC = false;
  constexpr bool HAS_COO = false;

  error::error_t status = cudaSuccess;

  // graph::graph_t<HAS_COO, HAS_CSR, HAS_CSC,
  //         vertex_t, edge_t, weight_t> graph;

  graph::graph_t<graph::graph_csr_t<vertex_t, edge_t, weight_t>> graph;

}

int
main(int argc, char** argv)
{
  test_graph();
  return;
}