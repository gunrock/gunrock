#include <gunrock/algorithms/pr.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;

void test_pr(int num_arguments, char** argument_array) {
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
                                        "Page Rank");

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

  // --
  // Params and memory allocation

  srand(time(NULL));

  weight_t alpha = 0.85;
  weight_t tol = 1e-6;

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> p(n_vertices);

  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(arguments.tag_string, &tag_vect);

  // --
  // GPU Run

  std::vector<float> run_times;

  auto benchmark_metrics =
      std::vector<benchmark::host_benchmark_t>(arguments.num_runs);
  for (int i = 0; i < arguments.num_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(gunrock::pr::run(G, alpha, tol, p.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Placeholder since PR does not use sources
  std::vector<int> src_placeholder;

  // Export metrics
  if (arguments.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "pr",
        arguments.filename, "market", arguments.json_dir, arguments.json_file,
        src_placeholder, tag_vect, num_arguments, argument_array);
  }

  // Log

  print::head(p, 40, "GPU rank");

  std::cout << "GPU Elapsed Time : " << run_times[arguments.num_runs - 1]
            << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_pr(argc, argv);
}
