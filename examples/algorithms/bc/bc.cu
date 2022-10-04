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

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Betweenness Centrality");

  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(mm.load(params.filename));
  }

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );  // supports row_indices and column_offsets (default = nullptr)

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> bc_values(n_vertices);
  int edges_visited = 0;
  int search_depth = 0;

  // Determine starting source
  vertex_t single_source;
  if (params.source == -1) {
    // Generate random starting source
    std::random_device seed;
    std::mt19937 engine(seed());
    std::uniform_int_distribution<int> dist(0, n_vertices - 1);
    single_source = dist(engine);
  } else if (params.source >= 0 && params.source < n_vertices) {
    single_source = params.source;
  } else {
    std::cout << "Error: invalid source"
              << "\n";
    exit(1);
  }

  // --
  // GPU Run

  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    // To alternatively compute for all vertices, call without single source
    // Collect run times without collecting metrics (due to overhead)
    run_times.push_back(gunrock::bc::run(G, single_source, false,
                                         bc_values.data().get(), &edges_visited,
                                         &search_depth));
  }

  // --
  // Log

  std::cout << "Single source : " << single_source << "\n";
  print::head(bc_values, 40, "GPU bc values");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // Run performance evaluation

  if (params.collect_metrics) {
    float metrics_run_time =
        gunrock::bc::run(G, single_source, params.collect_metrics,
                         bc_values.data().get(), &edges_visited, &search_depth);

    vertex_t n_edges = G.get_number_of_edges();

    // For BC - the number of nodes visited is just 2 * edges_visited
    gunrock::util::stats::get_performance_stats(
        edges_visited, (2 * edges_visited), n_edges, n_vertices, search_depth,
        run_times, "bc", params.filename, "market", params.json_dir,
        params.json_file, num_arguments, argument_array, GIT_SHA1);
  }
}

int main(int argc, char** argv) {
  test_bc(argc, argv);
}
