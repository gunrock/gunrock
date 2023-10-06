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

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Page Rank");

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

  srand(time(NULL));

  weight_t alpha = 0.85;
  weight_t tol = 1e-6;

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> p(n_vertices);

  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  std::vector<float> run_times;

#if !(ESSENTIALS_COLLECT_METRICS)
  // Standard run
  for (int i = 0; i < params.num_runs; i++) {
    run_times.push_back(gunrock::pr::run(G, alpha, tol, p.data().get()));
  }
  // Export metrics (runtimes only)
  if (params.export_metrics) {
    std::vector<int> empty_vector;
    gunrock::util::stats::export_performance_stats(
        empty_vector, empty_vector, n_edges, n_vertices, empty_vector,
        run_times, "pr", params.filename, "market", params.json_dir,
        params.json_file, empty_vector, tag_vect, num_arguments,
        argument_array);
  }
#else
  // Run with performance evaluation
  std::vector<int> edges_visited_vect(params.num_runs);
  std::vector<int> search_depth_vect(params.num_runs);
  std::vector<int> vertices_visited_vect(params.num_runs);

  for (int i = 0; i < params.num_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(gunrock::pr::run(G, alpha, tol, p.data().get()));

    thrust::host_vector<int> h_edges_visited = benchmark::____.edges_visited;
    thrust::host_vector<int> h_vertices_visited =
        benchmark::____.vertices_visited;

    edges_visited_vect[i] = h_edges_visited[0];
    vertices_visited_vect[i] = h_vertices_visited[0];
    search_depth_vect[i] = benchmark::____.search_depth;

    benchmark::DESTROY_BENCH();
  }

  // Placeholder since PR does not use sources
  std::vector<int> src_placeholder;

  // Export metrics
  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        edges_visited_vect, vertices_visited_vect, n_edges, n_vertices,
        search_depth_vect, run_times, "pr", params.filename, "market",
        params.json_dir, params.json_file, src_placeholder, tag_vect,
        num_arguments, argument_array);
  }
#endif

  // Log

  print::head(p, 40, "GPU rank");

  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_pr(argc, argv);
}
