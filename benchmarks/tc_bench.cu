#include <nvbench/nvbench.cuh>
#include <cxxopts.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/tc.hxx>

using namespace gunrock;
using namespace memory;

using vertex_t = uint32_t;
using edge_t = uint32_t;
using weight_t = float;
using count_t = vertex_t;

std::string filename_;
bool reduce_all_triangles_;
struct parameters_t {
  std::string filename;
  bool reduce_all_triangles;
  bool help = false;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv) : options(argv[0], "TC Benchmarking") {
    options.allow_unrecognised_options();
    // Add command line options
    options.add_options()("h,help", "Print help")(
        "m,market", "Matrix file", cxxopts::value<std::string>())(
        "r,reduce",
        "Compute a single triangle count for the entire graph (default = "
        "false)",
        cxxopts::value<bool>()->default_value("false"));

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
        reduce_all_triangles = result["reduce"].as<bool>();
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
    }
  }
};

void tc_bench(nvbench::state& state) {
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
  // Build graph + metadata
  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto mmatrix = mm.load(filename_);
  if (!mm_is_symmetric(mm.code)) {
    std::cerr << "Error: input matrix must be symmetric" << std::endl;
    exit(1);
  }
  csr.from_coo(mmatrix);

  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<vertex_t> column_indices(csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  auto G = graph::build::from_csr<memory_space_t::device,
                                  graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );

  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<count_t> triangles_count(n_vertices, 0);
  std::size_t total_triangles = 0;
  // --
  // Run TC with NVBench
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    tc::run(G, reduce_all_triangles_, triangles_count.data().get(),
            &total_triangles);
  });
}

int main(int argc, char** argv) {
  parameters_t params(argc, argv);
  filename_ = params.filename;
  reduce_all_triangles_ = params.reduce_all_triangles;

  if (params.help) {
    // Print NVBench help.
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Create a new argument array without TC options to pass to NVBench.
    char* args[argc];
    int j = 0;
    int num_tc_arguments = 0;
    for (int i = 0; i < argc; i++) {
      if (strcmp(argv[i], "--market") == 0 || strcmp(argv[i], "-m") == 0) {
        num_tc_arguments += 2;
        i++;
        continue;
      }
      if (strcmp(argv[i], "--reduce") == 0 || strcmp(argv[i], "-r") == 0) {
        num_tc_arguments += 1;
        continue;
      }
      args[j] = argv[i];
      j++;
    }
    NVBENCH_BENCH(tc_bench);
    NVBENCH_MAIN_BODY(argc - num_tc_arguments, args);
  }
}
