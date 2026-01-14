#include <gunrock/algorithms/bc.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;

void test_bc(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  gunrock::io::cli::parameters_t arguments(num_arguments, argument_array,
                                        "Betweenness Centrality");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(arguments.filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;

  if (arguments.binary) {
    csr.read_binary(arguments.filename);
  } else {
    csr.from_coo(coo);
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> bc_values(n_vertices);

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

    run_times.push_back(
        gunrock::bc::run(G, source_vect[i], bc_values.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (arguments.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "bc",
        arguments.filename, "market", arguments.json_dir, arguments.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // --
  // Log

  std::cout << "Single source : " << source_vect.back() << "\n";
  print::head(bc_values, 40, "GPU bc values");
  std::cout << "GPU Elapsed Time : " << run_times[arguments.num_runs - 1]
            << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_bc(argc, argv);
}
