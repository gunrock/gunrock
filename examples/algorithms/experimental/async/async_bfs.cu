#include <gunrock/algorithms/experimental/async/bfs.hxx>
#include "bfs_cpu.hxx"

using namespace gunrock;
using namespace experimental;
using namespace memory;

void test_async_bfs(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.from_coo(mm.load(filename));

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
  vertex_t single_source = 0;
  std::cout << "Single Source = " << single_source << std::endl;

  // --
  // GPU Run

  thrust::device_vector<vertex_t> depth(n_vertices);

  float gpu_elapsed = async::bfs::run(G, single_source, depth.data().get());
  cudaDeviceSynchronize();

  // --
  // CPU Run

  thrust::host_vector<vertex_t> h_depth(n_vertices);

  float cpu_elapsed =
      bfs_cpu::run<csr_t, vertex_t, edge_t>(csr, single_source, h_depth.data());

  int n_errors = util::compare(depth.data().get(), h_depth.data(), n_vertices);

  // --
  // Log + Validate
  print::head(depth, 40, "GPU depth");
  print::head(h_depth, 40, "CPU depth");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

int main(int argc, char** argv) {
  test_async_bfs(argc, argv);
  return EXIT_SUCCESS;
}
