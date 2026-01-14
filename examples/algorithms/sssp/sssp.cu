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

  // --
  // Params and memory allocation

  srand(time(NULL));

  vertex_t n_vertices = G.get_number_of_vertices();

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  thrust::device_vector<int> vertices_visited(1);
  int search_depth = 0;

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(arguments.source_string, &source_vect,
                                        n_vertices, arguments.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(arguments.tag_string, &tag_vect);

  // --
  // GPU Run

  /// An example of how one can use std::shared_ptr to allocate memory on the
  /// GPU, using a custom deleter that automatically handles deletion of the
  /// memory.
  // std::shared_ptr<weight_t> distances(
  //     allocate<weight_t>(n_vertices * sizeof(weight_t)),
  //     deleter_t<weight_t>());
  // std::shared_ptr<vertex_t> predecessors(
  //     allocate<vertex_t>(n_vertices * sizeof(vertex_t)),
  //     deleter_t<vertex_t>());

  std::vector<float> run_times;
  for (int i = 0; i < source_vect.size(); i++) {
    // Use new run API with param_t
    gunrock::sssp::param_t<vertex_t> param(
        source_vect[i], 
        false,  // collect_metrics
        arguments.advance_load_balance,
        arguments.filter_algorithm,
        arguments.enable_filter);
    gunrock::sssp::result_t<vertex_t, weight_t> result(
        distances.data().get(),
        predecessors.data().get(),
        edges_visited.data().get(),
        vertices_visited.data().get(),
        &search_depth,
        n_vertices);
    
    run_times.push_back(gunrock::sssp::run(G, param, result));
  }

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

  // --
  // Run performance evaluation

  if (arguments.collect_metrics) {
    std::vector<int> edges_visited_vect;
    std::vector<int> search_depth_vect;
    std::vector<int> nodes_visited_vect;

    vertex_t n_edges = G.get_number_of_edges();

    for (int i = 0; i < source_vect.size(); i++) {
      // Use new run API with param_t for metrics collection
      gunrock::sssp::param_t<vertex_t> param(
          source_vect[i], 
          true,  // collect_metrics
          arguments.advance_load_balance,
          arguments.filter_algorithm,
          arguments.enable_filter);
      gunrock::sssp::result_t<vertex_t, weight_t> result(
          distances.data().get(),
          predecessors.data().get(),
          edges_visited.data().get(),
          vertices_visited.data().get(),
          &search_depth,
          n_vertices);
      
      float metrics_run_time = gunrock::sssp::run(G, param, result);

      thrust::host_vector<int> h_edges_visited = edges_visited;
      thrust::host_vector<int> h_vertices_visited = vertices_visited;

      edges_visited_vect.push_back(h_edges_visited[0]);
      nodes_visited_vect.push_back(h_vertices_visited[0]);
      search_depth_vect.push_back(search_depth);
    }

    gunrock::util::stats::get_performance_stats(
        edges_visited_vect, nodes_visited_vect, n_edges, n_vertices,
        search_depth_vect, run_times, "sssp", arguments.filename, "market",
        arguments.json_dir, arguments.json_file, source_vect, tag_vect, num_arguments,
        argument_array);
  }
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
