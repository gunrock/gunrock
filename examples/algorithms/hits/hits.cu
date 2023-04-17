#include <gunrock/algorithms/hits.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;

void test_hits(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Betweenness Centrality");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  gunrock::io::loader_struct<vertex_t, edge_t, weight_t> loader;
  loader = mm.load(params.filename);
  
  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(loader.coo);
  }

  // --
  // Build graph

  auto G =
      graph::build::build<memory_space_t::device>(loader.properties, csr);

  hits::result_c<vertex_t, weight_t> result;
  unsigned int max_iter = 20;

  // --
  // GPU Run

  auto time = gunrock::hits::run(G, max_iter, result);
  result.print_result();
}

int main(int argc, char** argv) {
  test_hits(argc, argv);
  return 0;
}
