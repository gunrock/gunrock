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

  vertex_t n_vertices = G.get_number_of_vertices();
  vertex_t n_edges = G.get_number_of_edges();
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

  uint n_runs = source_vect.size();
  std::vector<float> run_times;

#if !(ESSENTIALS_COLLECT_METRICS) 
  // Standard run
  for (int i = 0; i < n_runs; i++) {
    run_times.push_back(gunrock::bfs::run(
        G, source_vect[i], distances.data().get(), predecessors.data().get()));
  }
#else
  // Run with performance evaluation
  std::vector<int> edges_visited_vect(n_runs);
  std::vector<int> search_depth_vect(n_runs);
  std::vector<int> vertices_visited_vect(n_runs);

  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(gunrock::bfs::run(
        G, source_vect[i], distances.data().get(), predecessors.data().get()));

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
      search_depth_vect, run_times, "bfs", params.filename, "market",
      params.json_dir, params.json_file, source_vect, tag_vect, 
      num_arguments, argument_array);
#endif

  // Print info for last run
  std::cout << "Source : " << source_vect.back() << "\n";
  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[n_runs - 1] << " (ms)"
            << std::endl;

  // --
  // CPU Run

  if (params.validate) {
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
