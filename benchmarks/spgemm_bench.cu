#include <nvbench/nvbench.cuh>
#include <cxxopts.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/spgemm.hxx>

#include "benchmarks.hxx"

using namespace gunrock;
using namespace memory;

using vertex_t = int;
using edge_t = int;
using weight_t = float;

std::string filename_a;
std::string filename_b;

struct parameters_t {
  std::string filename_a;
  std::string filename_b;
  bool help = false;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "SPGEMM Benchmarking") {
    options.allow_unrecognised_options();
    // Add command line options
    options.add_options()("h,help", "Print help")  // help
        ("a,amatrix", "Matrix A file",
         cxxopts::value<std::string>())  // mtx A
        ("b,bmatrix", "Matrix B file",
         cxxopts::value<std::string>());  // mtx B

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      help = true;
      std::cout << options.help({""});
      std::cout << "  [optional nvbench args]" << std::endl << std::endl;
      // Do not exit so we also print NVBench help.
    } else {
      if (result.count("amatrix") == 1) {
        filename_a = result["amatrix"].as<std::string>();
        if (!util::is_market(filename_a)) {
          std::cout << options.help({""});
          std::cout << "  [optional nvbench args]" << std::endl << std::endl;
          std::exit(0);
        }
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
      if (result.count("bmatrix") == 1) {
        filename_b = result["bmatrix"].as<std::string>();
        if (!util::is_market(filename_b)) {
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

void spgemm_bench(nvbench::state& state) {
  // --
  // Add metrics
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  // --
  // Define types
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  using csc_t =
      format::csc_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // Build graphs + metadata
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  csr_t a_csr;
  auto [a_properties, a_coo] = mm.load(filename_a);
  a_csr.from_coo(a_coo);

  auto A = graph::build<memory_space_t::device>(a_properties, a_csr);

  csr_t b_csr;
  csc_t b_csc;

  auto [b_properties, b_coo] = mm.load(filename_b);

  b_csr.from_coo(b_coo);
  b_csc.from_csr(b_csr);

  auto B = graph::build<memory_space_t::device>(b_properties, b_csc, b_csr);

  csr_t C;

  // --
  // Run SPGEMM with NVBench
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { gunrock::spgemm::run(A, B, C); });
}

int main(int argc, char** argv) {
  parameters_t params(argc, argv);
  filename_a = params.filename_a;
  filename_b = params.filename_b;

  if (params.help) {
    // Print NVBench help.
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Remove all gunrock parameters and pass to nvbench.
    auto args = filtered_argv(argc, argv, "--amatrix", "-a", "--bmatrix", "-b",
                              filename_a, filename_b);
    NVBENCH_BENCH(spgemm_bench);
    NVBENCH_MAIN_BODY(args.size(), args.data());
  }
}
