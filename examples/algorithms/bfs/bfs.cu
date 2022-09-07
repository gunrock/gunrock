#include <gunrock/algorithms/bfs.hxx>
#include "bfs_cpu.hxx"  // Reference implementation
#include <sys/utsname.h>
#include "gunrock/util/performance.hxx"
#include <cxxopts.hpp>

using namespace gunrock;
using namespace memory;

struct parameters_t {
  std::string filename;
  std::string json = "";
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
    options.add_options()("help", "Print help")                     // help
        ("validate", "CPU validation")                              // validate
        ("performance", "performance analysis")                     // validate
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx
        ("j,json", "JSON output file",
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

    // TODO: check for valid file path
    if (result.count("json") == 1) {
      json = result["json"].as<std::string>();
    }
  }
};

void test_bfs(int num_arguments, char** argument_array) {
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
  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    std::cout << params.filename << "\n";
    csr.from_coo(mm.load(params.filename));
  }

  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<vertex_t> column_indices(csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  // --
  // Build graph + metadata

  auto G =
      graph::build::from_csr<memory_space_t::device,
                             graph::view_t::csr /* | graph::view_t::csc */>(
          csr.number_of_rows,               // rows
          csr.number_of_columns,            // columns
          csr.number_of_nonzeros,           // nonzeros
          csr.row_offsets.data().get(),     // row_offsets
          csr.column_indices.data().get(),  // column_indices
          csr.nonzero_values.data().get(),  // values
          row_indices.data().get(),         // row_indices
          column_offsets.data().get()       // column_offsets
      );

  // --
  // Params and memory allocation

  vertex_t single_source = 0;

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  thrust::device_vector<int> search_depth(1);

  // --
  // Run problem

  float gpu_elapsed = gunrock::bfs::run(
      G, single_source, distances.data().get(), predecessors.data().get(),
      edges_visited.data().get(), search_depth.data().get());

  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    thrust::host_vector<vertex_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
        csr, single_source, h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);
    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }

  if (params.performance) {
    thrust::host_vector<int> h_edges_visited = edges_visited;
    thrust::host_vector<int> h_search_depth = search_depth;

    // For BFS - the number of nodes visited is just 2 * edges_visited
    get_performance_stats(h_edges_visited[0], (2 * h_edges_visited[0]),
                          h_search_depth[0], gpu_elapsed, "bfs",
                          params.filename, "market", params.json);
  }
}

int main(int argc, char** argv) {
  test_bfs(argc, argv);
}
