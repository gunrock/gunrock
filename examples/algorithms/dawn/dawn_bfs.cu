#include <gunrock/algorithms/dawn.hxx>
#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/framework/benchmark.hxx>

using namespace gunrock;
using namespace memory;

void test_dawn_bfs(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Breadth First Search");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);

  csr_t csr;

  if (params.binary) {
    csr.read_binary(params.filename);
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
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // Run problem

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;
  std::vector<float> run_times_bfs;

  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(gunrock::dawn_bfs::run(
        G, source_vect[i], distances.data().get(), predecessors.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "dawn_bfs",
        params.filename, "market", params.json_dir, params.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // Print info for last run
  std::cout << "Source : " << source_vect.back() << "\n";
  print::head(distances, 40, "GPU distances");
  std::cout << "[DAWN BFS Advance] GPU Elapsed Time : " << run_times[n_runs - 1]
            << " (ms)" << std::endl;

  auto benchmark_metrics_bfs = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    run_times_bfs.push_back(gunrock::bfs::run(
        G, source_vect[i], distances.data().get(), predecessors.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics_bfs[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics_bfs, n_edges, n_vertices, run_times_bfs, "bfs",
        params.filename, "market", params.json_dir, params.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // Print info for last run
  std::cout << "Source : " << source_vect.back() << "\n";
  print::head(distances, 40, "GPU distances");
  std::cout << "[BFS] GPU Elapsed Time : " << run_times_bfs[n_runs - 1]
            << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_dawn_bfs(argc, argv);
}
