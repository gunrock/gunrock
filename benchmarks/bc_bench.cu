#include <nvbench/nvbench.cuh>
#include <cxxopts.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/bc.hxx>

#include "benchmarks.hxx"

using namespace gunrock;
using namespace memory;

using vertex_t = int;
using edge_t = int;
using weight_t = float;

std::string filename;

struct parameters_t {
  std::string filename;
  bool help = false;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv) : options(argv[0], "BC Benchmarking") {
    options.allow_unrecognised_options();
    // Add command line options
    options.add_options()("h,help", "Print help")  // help
        ("m,market", "Matrix file",
         cxxopts::value<std::string>());  // mtx

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      help = true;
      std::cout << options.help({""});
      std::cout << "  [optional nvbench args]" << std::endl << std::endl;
      // Do not exit so we also print NVBench help.
    } else {
      if (result.count("market") == 1) {
        filename = result["market"].as<std::string>();
        if (!util::is_market(filename)) {
          std::cout << options.help({""});
          std::cout << "  [optional nvbench args]" << std::endl << std::endl;
          std::exit(0);
        }
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
    }
  }
};

void bc_bench(nvbench::state& state) {
  // --
  // Add metrics
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  // --
  // Build graph + metadata
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(coo);

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> bc_values(n_vertices);
  int edges_visited = 0;
  int search_depth = 0;

  // --
  // Run BC with NVBench
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    gunrock::bc::run(G, false, bc_values.data().get(), &edges_visited,
                     &search_depth);
  });
}

int main(int argc, char** argv) {
  parameters_t params(argc, argv);
  filename = params.filename;

  if (params.help) {
    // Print NVBench help.
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Remove all gunrock parameters and pass to nvbench.
    auto args = filtered_argv(argc, argv, "--market", "-m", filename);
    NVBENCH_BENCH(bc_bench);
    NVBENCH_MAIN_BODY(args.size(), args.data());
  }
}
