#include <nvbench/nvbench.cuh>
#include <cxxopts.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/spgemm.hxx>

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

  // --
  // Build graphs + metadata
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  csr_t a_csr;
  a_csr.from_coo(mm.load(filename_a));

  auto A = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      a_csr.number_of_rows, a_csr.number_of_columns, a_csr.number_of_nonzeros,
      a_csr.row_offsets.data().get(), a_csr.column_indices.data().get(),
      a_csr.nonzero_values.data().get());

  csr_t b_csr;
  b_csr.from_coo(mm.load(filename_b));

  auto B = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      b_csr.number_of_rows, b_csr.number_of_columns, b_csr.number_of_nonzeros,
      b_csr.row_offsets.data().get(), b_csr.column_indices.data().get(),
      b_csr.nonzero_values.data().get());

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
    // Create a new argument array without matrix filenames to pass to NVBench.
    char* args[argc - 4];
    int j = 0;
    for (int i = 0; i < argc; i++) {
      if (strcmp(argv[i], "--amatrix") == 0 || strcmp(argv[i], "-a") == 0 ||
          strcmp(argv[i], "--bmatrix") == 0 || strcmp(argv[i], "-b") == 0) {
        i++;
        continue;
      }
      args[j] = argv[i];
      j++;
    }

    NVBENCH_BENCH(spgemm_bench);
    NVBENCH_MAIN_BODY(argc - 4, args);
  }
}
