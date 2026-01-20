#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/framework/benchmark.hxx>

#include "bfs_cpu.hxx"  // Reference implementation

using namespace gunrock;
using namespace memory;

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

  gunrock::io::cli::parameters_t arguments(num_arguments, argument_array,
                                        "Breadth First Search");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(arguments.filename);

  csr_t csr;

  if (arguments.binary) {
    csr.read_binary(arguments.filename);
  } else {
    csr.from_coo(coo);
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // Create context
  auto context = std::make_shared<gcuda::multi_context_t>(0);

  // --
  // Params and memory allocation

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(arguments.source_string, &source_vect,
                                        n_vertices, arguments.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(arguments.tag_string, &tag_vect);

  // --
  // Run problem

  // Get optimization options from CLI arguments
  gunrock::options_t options = arguments.get_options();

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;

  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    // Synchronize before each run to ensure clean state
    // This is critical for multiple runs to prevent segfaults
    context->get_context(0)->synchronize();
    
    benchmark::INIT_BENCH();

    // Create param and result structs with CLI options
    gunrock::bfs::param_t<vertex_t> param(source_vect[i], options);
    gunrock::bfs::result_t<vertex_t> result(distances.data().get(),
                                            predecessors.data().get());

    run_times.push_back(gunrock::bfs::run(G, param, result, context));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
    
    // Synchronize after each run to ensure all operations complete
    context->get_context(0)->synchronize();
  }

  // Export metrics
  if (arguments.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "bfs",
        arguments.filename, "market", arguments.json_dir, arguments.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // Print info for last run
  std::cout << "Source : " << source_vect.back() << "\n";
  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[n_runs - 1] << " (ms)"
            << std::endl;

  // --
  // CPU Run

  if (arguments.validate) {
    thrust::host_vector<vertex_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    // Validate with last source in source vector
    float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
        csr, source_vect.back(), h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);
    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }
}

int main(int argc, char** argv) {
  test_bfs(argc, argv);
}
