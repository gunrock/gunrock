std::string filename;

#include "algorithms/mst_bench.cu"
#include "algorithms/bfs_bench.cu"
#include "algorithms/bc_bench.cu"
#include "algorithms/color_bench.cu"
#include "algorithms/kcore_bench.cu"
#include "algorithms/ppr_bench.cu"
#include "algorithms/pr_bench.cu"
#include "algorithms/spmv_bench.cu"
#include "algorithms/sssp_bench.cu"

std::vector<std::string> benchmarks;
struct parameters_t {
  std::string filename;
  std::string benchmark;
  bool help = false;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Algorithm Benchmarks") {
    options.allow_unrecognised_options();
    // Add command line options
    options.add_options()("h,help", "Print help")  // help
        ("m,market", "Matrix file (required)",
         cxxopts::value<std::string>())  // mtx
        ("b,benchmark", "Benchmark name (optional)",
         cxxopts::value<std::string>());  // benchmark

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      help = true;
      std::cout << options.help({""});
      std::cout << "  [optional nvbench args]" << std::endl << std::endl;
    } else {
      if (result.count("market") == 1) {
        filename = result["market"].as<std::string>();
        if (util::is_market(filename)) {
        } else {
          std::cout << options.help({""});
          std::cout << "  [optional nvbench args]" << std::endl << std::endl;
          std::exit(0);
        }
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }

      if (result.count("benchmark") == 1) {
        benchmark = result["benchmark"].as<std::string>();
        if (std::find(benchmarks.begin(), benchmarks.end(), benchmark) ==
            benchmarks.end()) {
          std::cout << "Error: invalid benchmark" << std::endl;
          std::exit(0);
        }
      } else {
        benchmark = "all";
      }
    }
  }
};

int main(int argc, char** argv) {
  benchmarks = {"mst_bench",   "bfs_bench",   "bc_bench",
                "color_bench", "kcore_bench", "ppr_bench",
                "pr_bench",    "spmv_bench",  "sssp_bench"};

  parameters_t params(argc, argv);
  filename = params.filename;
  std::string benchmark = params.benchmark;

  if (params.help) {
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Create a new argument array without matrix filename to pass to NVBench.
    char* args[argc - 2];
    int j = 0;
    for (int i = 0; i < argc; i++) {
      if (strcmp(argv[i], "--market") == 0 || strcmp(argv[i], "-m") == 0) {
        i++;
        continue;
      }
      args[j] = argv[i];
      j++;
    }

    NVBENCH_BENCH(mst_bench);
    NVBENCH_BENCH(bfs_bench);
    NVBENCH_BENCH(bc_bench);
    NVBENCH_BENCH(color_bench);
    NVBENCH_BENCH(kcore_bench);
    NVBENCH_BENCH(ppr_bench);
    NVBENCH_BENCH(pr_bench);
    NVBENCH_BENCH(spmv_bench);
    NVBENCH_BENCH(sssp_bench);
    NVBENCH_MAIN_BODY(argc - 2, args);
  }
}