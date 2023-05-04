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

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> p(n_vertices);
  int edges_visited = 0;
  int search_depth = 0;

  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    // Record run times without collecting metrics (due to overhead)
    run_times.push_back(
        gunrock::pr::run(G, alpha, tol, false, p.data().get(), &search_depth));
  }

  // --
  // Log

  print::head(p, 40, "GPU rank");

  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // Run performance evaluation

  if (params.collect_metrics) {
    std::vector<int> edges_visited_vect;
    std::vector<int> nodes_visited_vect;
    std::vector<int> search_depth_vect;

    vertex_t n_edges = G.get_number_of_edges();

    for (int i = 0; i < params.num_runs; i++) {
      float metrics_run_time = gunrock::pr::run(
          G, alpha, tol, params.collect_metrics, p.data().get(), &search_depth);
      search_depth_vect.push_back(search_depth);
    }
    // For PR - we visit every edge in the graph during each iteration
    edges_visited = n_edges * (search_depth + 1);

    edges_visited_vect.insert(edges_visited_vect.end(), params.num_runs,
                              edges_visited);
    // For PR - the number of nodes visited is just 2 * edges_visited
    nodes_visited_vect.insert(nodes_visited_vect.end(), params.num_runs,
                              2 * edges_visited);

    // Placeholder since PR does not use sources
    std::vector<int> src_placeholder;

    gunrock::util::stats::get_performance_stats(
        edges_visited_vect, nodes_visited_vect, n_edges, n_vertices,
        search_depth_vect, run_times, "pr", params.filename, "market",
        params.json_dir, params.json_file, src_placeholder, tag_vect,
        num_arguments, argument_array);
  }
}

int main(int argc, char** argv) {
  test_pr(argc, argv);
}
