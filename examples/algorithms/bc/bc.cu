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

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> bc_values(n_vertices);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run
  
  uint n_runs = source_vect.size();
  std::vector<float> run_times;

#if !(ESSENTIALS_COLLECT_METRICS) 
  // Standard run
  for (int i = 0; i < n_runs; i++) {
    run_times.push_back(gunrock::bc::run(G, source_vect[i],
                                         bc_values.data().get()));
  }
#else
  // Run with performance evaluation
  std::vector<int> edges_visited_vect(n_runs);
  std::vector<int> search_depth_vect(n_runs);
  std::vector<int> vertices_visited_vect(n_runs);

  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(gunrock::bc::run(G, source_vect[i],
                                         bc_values.data().get()));

    thrust::host_vector<int> h_edges_visited = benchmark::____.edges_visited;
    thrust::host_vector<int> h_vertices_visited =
        benchmark::____.vertices_visited;

    edges_visited_vect[i] = h_edges_visited[0];
    vertices_visited_vect[i] = h_vertices_visited[0];
    search_depth_vect[i] = benchmark::____.search_depth;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  gunrock::util::stats::get_performance_stats(
      edges_visited_vect, vertices_visited_vect, n_edges, n_vertices,
      search_depth_vect, run_times, "bc", params.filename, "market",
      params.json_dir, params.json_file, source_vect, tag_vect, 
      num_arguments, argument_array);
#endif

  // --
  // Log

  std::cout << "Single source : " << source_vect.back() << "\n";
  print::head(bc_values, 40, "GPU bc values");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_bc(argc, argv);
}
