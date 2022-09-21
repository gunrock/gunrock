#include <gunrock/algorithms/sssp.hxx>
#include "sssp_cpu.hxx"  // Reference implementation
#include "gunrock/util/performance.hxx"
#include <cxxopts.hpp>

using namespace gunrock;
using namespace memory;

struct parameters_t {
  std::string filename;
  std::string json_dir = ".";
  std::string json_file = "";
  int num_runs = 1;
  cxxopts::Options options;
  bool validate = false;
  bool performance = false;
  bool binary = false;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Breadth First Search example") {
    // Add command line options
    options.add_options()("help", "Print help")  // help
        ("validate", "CPU validation")           // validate
        ("performance", "performance analysis")  // performance evaluation
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx file
        ("n,num_runs", "Number of runs", cxxopts::value<int>())     // runs
        ("d,json_dir", "JSON output directory",
         cxxopts::value<std::string>())  // json output directory
        ("f,json_file", "JSON output file",
         cxxopts::value<std::string>());  // json output file

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") == 1) {
      filename = result["market"].as<std::string>();
      if (util::is_binary_csr(filename)) {
        binary = true;
      } else if (!util::is_market(filename)) {
        std::cout << options.help({""}) << std::endl;
        std::exit(0);
      }
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("validate") == 1) {
      validate = true;
    }

    if (result.count("performance") == 1) {
      performance = true;
    }

    if (result.count("num_runs") == 1) {
      num_runs = result["num_runs"].as<int>();
    }

    if (result.count("json_dir") == 1) {
      json_dir = result["json_dir"].as<std::string>();
    }

    if (result.count("json_file") == 1) {
      json_dir = result["json_file"].as<std::string>();
    }
  }
};

void test_sssp(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  parameters_t params(num_arguments, argument_array);

  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(mm.load(params.filename));
  }

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );  // supports row_indices and column_offsets (default = nullptr)

  // --
  // Params and memory allocation
  srand(time(NULL));

  vertex_t n_vertices = G.get_number_of_vertices();
  vertex_t single_source = 0;  // rand() % n_vertices;
  std::cout << "Single Source = " << single_source << std::endl;

  // --
  // GPU Run

  /// An example of how one can use std::shared_ptr to allocate memory on the
  /// GPU, using a custom deleter that automatically handles deletion of the
  /// memory.
  // std::shared_ptr<weight_t> distances(
  //     allocate<weight_t>(n_vertices * sizeof(weight_t)),
  //     deleter_t<weight_t>());
  // std::shared_ptr<vertex_t> predecessors(
  //     allocate<vertex_t>(n_vertices * sizeof(vertex_t)),
  //     deleter_t<vertex_t>());

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  thrust::device_vector<int> vertices_visited(1);
  int search_depth = 0;

  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    run_times.push_back(gunrock::sssp::run(
        G, single_source, params.performance, distances.data().get(),
        predecessors.data().get(), edges_visited.data().get(),
        vertices_visited.data().get(), &search_depth));
  }

  // --
  // CPU Run

  thrust::host_vector<weight_t> h_distances(n_vertices);
  thrust::host_vector<vertex_t> h_predecessors(n_vertices);

  float cpu_elapsed = sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
      csr, single_source, h_distances.data(), h_predecessors.data());

  int n_errors =
      util::compare(distances.data().get(), h_distances.data(), n_vertices);

  // --
  // Log + Validate

  print::head(distances, 40, "GPU distances");
  print::head(h_distances, 40, "CPU Distances");

  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;

  // --
  // Run performance evaluation

  if (params.performance) {
    vertex_t n_edges = G.get_number_of_edges();

    // For BFS - the number of nodes visited is just 2 * edges_visited
    thrust::host_vector<int> h_edges_visited = edges_visited;
    thrust::host_vector<int> h_vertices_visited = vertices_visited;
    get_performance_stats(h_edges_visited[0], h_vertices_visited[0], n_edges,
                          n_vertices, search_depth, run_times, "sssp",
                          params.filename, "market", params.json_dir,
                          params.json_file);
  }
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
