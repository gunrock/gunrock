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

  srand(time(NULL));

  weight_t alpha = 0.85;
  weight_t tol = 1e-6;

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> p(n_vertices);
  int edges_visited = 0;
  int search_depth = 0;

  // --
  // GPU Run

  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    run_times.push_back(gunrock::pr::run(G, alpha, tol, params.collect_metrics,
                                         p.data().get(), &search_depth));
  }

  // --
  // Log

  print::head(p, 40, "GPU rank");

  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;

  // --
  // Run performance evaluation

  if (params.collect_metrics) {
    vertex_t n_edges = G.get_number_of_edges();
    // For PR - we visit every edge in the graph during each iteration
    edges_visited = n_edges * (search_depth + 1);

    // For PR - the number of nodes visited is just 2 * edges_visited
    gunrock::util::stats::get_performance_stats(
        edges_visited, (2 * edges_visited), n_edges, n_vertices, search_depth,
        run_times, "pr", params.filename, "market", params.json_dir,
        params.json_file);
  }
}

int main(int argc, char** argv) {
  test_pr(argc, argv);
}
