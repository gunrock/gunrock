#include <gunrock/algorithms/sssp.hxx>
#include "sssp_cpu.hxx"  // Reference implementation
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;

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

  gunrock::io::cli::parameters_t arguments(num_arguments, argument_array,
                                        "Single Source Shortest Path");

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

  srand(time(NULL));

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  thrust::device_vector<int> vertices_visited(1);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(arguments.source_string, &source_vect,
                                        n_vertices, arguments.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(arguments.tag_string, &tag_vect);

  // --
  // GPU Run

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;

  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    // Use legacy API to test
    run_times.push_back(gunrock::sssp::run(G, source_vect[i],
                                           distances.data().get(),
                                           predecessors.data().get(),
                                           context));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (arguments.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "sssp",
        arguments.filename, "market", arguments.json_dir, arguments.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // --
  // Log

  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[arguments.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // CPU Run

  if (arguments.validate) {
    thrust::host_vector<weight_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    float cpu_elapsed = sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
        csr, source_vect.back(), h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);

    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
