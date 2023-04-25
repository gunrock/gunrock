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

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Betweenness Centrality");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;

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

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> bc_values(n_vertices);
  int edges_visited = 0;
  int search_depth = 0;

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  std::vector<float> run_times;
  for (int i = 0; i < source_vect.size(); i++) {
    // Record run times without collecting metrics (due to overhead)
    run_times.push_back(gunrock::bc::run(G, source_vect[i], true,
                                         bc_values.data().get(), &edges_visited,
                                         &search_depth));
  }

  // --
  // Log

  std::cout << "Single source : " << source_vect.back() << "\n";
  print::head(bc_values, 40, "GPU bc values");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // Run performance evaluation

  if (params.collect_metrics) {
    std::vector<int> edges_visited_vect;
    std::vector<int> search_depth_vect;
    std::vector<int> nodes_visited_vect(source_vect.size());

    vertex_t n_edges = G.get_number_of_edges();

    for (int i = 0; i < source_vect.size(); i++) {
      float metrics_run_time = gunrock::bc::run(
          G, source_vect[i], params.collect_metrics, bc_values.data().get(),
          &edges_visited, &search_depth);

      edges_visited_vect.push_back(edges_visited);
      search_depth_vect.push_back(search_depth);
    }

    // For BC - the number of nodes visited is just 2 * edges_visited
    std::transform(edges_visited_vect.begin(), edges_visited_vect.end(),
                   nodes_visited_vect.begin(), [](auto& c) { return 2 * c; });

    gunrock::util::stats::get_performance_stats(
        edges_visited_vect, nodes_visited_vect, n_edges, n_vertices,
        search_depth_vect, run_times, "bc", params.filename, "market",
        params.json_dir, params.json_file, source_vect, tag_vect, num_arguments,
        argument_array);
  }
}

int main(int argc, char** argv) {
  test_bc(argc, argv);
}
