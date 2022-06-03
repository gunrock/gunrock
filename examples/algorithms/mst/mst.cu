#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/mst.hxx>
#include "mst_cpu.hxx"  // Reference implementation
#include <cxxopts.hpp>
#include <iomanip>

using namespace gunrock;
using namespace memory;

struct parameters_t {
  std::string filename;
  cxxopts::Options options;
  bool validate;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Minimum Spanning Tree example") {
    // Add command line options
    options.add_options()("help", "Print help")                      // help
        ("validate", "CPU validation")                               // validate
        ("m,market", "Matrix file", cxxopts::value<std::string>());  // mtx

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") == 1) {
      filename = result["market"].as<std::string>();
      if (util::is_market(filename)) {
      } else {
        std::cout << options.help({""}) << std::endl;
        std::exit(0);
      }
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("validate") == 1) {
      validate = true;
    } else {
      validate = false;
    }
  }
};

void test_mst(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
  parameters_t params(num_arguments, argument_array);

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto mmatrix = mm.load(params.filename);
  if (!mm_is_symmetric(mm.code)) {
    printf("Error: input matrix must be symmetric\n");
    exit(1);
  }
  csr.from_coo(mmatrix);

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> mst_weight(1);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::mst::run(G, mst_weight.data().get());
  thrust::host_vector<weight_t> h_mst_weight = mst_weight;
  std::cout << "GPU MST Weight: " << std::fixed << std::setprecision(4)
            << h_mst_weight[0] << std::endl;
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    weight_t cpu_mst_weight;
    float cpu_elapsed =
        mst_cpu::run<csr_t, vertex_t, edge_t, weight_t>(csr, &cpu_mst_weight);
    std::cout << "CPU MST Weight: " << std::fixed << std::setprecision(4)
              << cpu_mst_weight << std::endl;
    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  }
}

int main(int argc, char** argv) {
  test_mst(argc, argv);
}