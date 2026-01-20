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

  // --
  // IO

  gunrock::io::cli::parameters_t arguments(num_arguments, argument_array,
                                        "Hyperlink-Induced Topic Search");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(arguments.filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;

  if (arguments.binary) {
    csr.read_binary(arguments.filename);
  } else {
    csr.from_coo(coo);
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  hits::result_c<vertex_t, weight_t> result;
  unsigned int max_iter = 20;

  // Get optimization options from CLI arguments
  gunrock::options_t options = arguments.get_options();

  // --
  // GPU Run

  // Create param struct with CLI options
  gunrock::hits::param_t param(max_iter, options);

  auto time = gunrock::hits::run(G, max_iter, result);
  result.print_result();

  std::cout << "GPU Elapsed Time : " << time << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_hits(argc, argv);
  return 0;
}
