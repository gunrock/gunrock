#include <gunrock/algorithms/bfs.hxx>
#include "bfs_cpu.hxx"  // Reference implementation
#include <sys/utsname.h>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>

using namespace gunrock;
using namespace memory;

#include <random>

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

  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(mm.load(params.filename));
  }

  // --
  // Build graph + metadata

  auto G =
      graph::build::from_csr<memory_space_t::device,
                             graph::view_t::csr /* | graph::view_t::csc */>(
          csr.number_of_rows,               // rows
          csr.number_of_columns,            // columns
          csr.number_of_nonzeros,           // nonzeros
          csr.row_offsets.data().get(),     // row_offsets
          csr.column_indices.data().get(),  // column_indices
          csr.nonzero_values.data().get()  // values
      );

  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();

  std::random_device rd; // obtain a random number from hardware
  auto current_state = rd();
  std::mt19937 gen(current_state); // seed the generator
  std::uniform_int_distribution<> distr(0, n_vertices); // define the range
      
  vertex_t single_source = distr(gen);

  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);
  thrust::device_vector<int> edges_visited(1);
  int search_depth = 0;

  // --
  // Run problem

  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    run_times.push_back(gunrock::bfs::run(
        G, single_source, params.collect_metrics, distances.data().get(),
        predecessors.data().get(), edges_visited.data().get(), &search_depth));
  }

  // print::head(distances, 40, "GPU distances");
  // std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
  //           << " (ms)" << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    thrust::host_vector<vertex_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
        csr, single_source, h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);
    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }

  // --
  // Run performance evaluation

  if (params.collect_metrics) {
    thrust::host_vector<int> h_edges_visited = edges_visited;
    vertex_t n_edges = G.get_number_of_edges();

    // For BFS - the number of nodes visited is just 2 * edges_visited
    gunrock::util::stats::get_performance_stats(current_state,
        h_edges_visited[0], (2 * h_edges_visited[0]), n_edges, n_vertices,
        search_depth, run_times, "bfs", params.filename, "market",
        params.json_dir, params.json_file);
  }
}

int main(int argc, char** argv) {
  test_bfs(argc, argv);
}
