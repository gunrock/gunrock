#include <cstdlib>  // EXIT_SUCCESS
#include <gunrock/applications/sssp.hxx>
#include "sssp_cpu.hxx"  // Reference implementation

using namespace gunrock;
using namespace memory;

void test_sssp(int num_arguments, char** argument_array) {
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
      format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t>;
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

  vertex_t single_source = 0;

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::sssp::run(
      G, single_source, distances.data().get(), predecessors.data().get());

  // --
  // CPU Run

  thrust::host_vector<weight_t> h_distances(n_vertices);
  thrust::host_vector<vertex_t> h_predecessors(n_vertices);

  float cpu_elapsed = sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
      csr, single_source, h_distances.data(), h_predecessors.data());

  int n_errors = sssp_cpu::compute_error(distances, h_distances);

  // --
  // Log + Validate

  std::cout << "GPU Distances (output) = ";
  thrust::copy(distances.begin(), distances.end(),
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "CPU Distances (output) = ";
  thrust::copy(h_distances.begin(), h_distances.end(),
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
  return EXIT_SUCCESS;
}
