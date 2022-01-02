#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/kcore.hxx>
#include "kcore_cpu.hxx"

using namespace gunrock;
using namespace memory;

void test_kcore(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types
  // Specify the types that will be used for
  // - vertex ids (vertex_t)
  // - edge offsets (edge_t)
  // - edge weights (weight_t)

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // --
  // IO

  // Filename to be read
  std::string filename = argument_array[1];

  // Load the matrix-market dataset into csr format.
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph

  // Convert the dataset you loaded into an `essentials` graph.
  // `memory_space_t::device` -> the graph will be created on the GPU.
  // `graph::view_t::csr`     -> your input data is in `csr` format.
  //
  // Note that `graph::build::from_csr` expects pointers, but the `csr` data
  // arrays are `thrust` vectors, so we need to unwrap them w/ `.data().get()`.
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  std::cout << "G.get_number_of_vertices() : " << G.get_number_of_vertices()
            << std::endl;
  std::cout << "G.get_number_of_edges()    : " << G.get_number_of_edges()
            << std::endl;
  std::cout << "G.is_directed()    : " << G.is_directed() << std::endl;

  // --
  // Params and memory allocation

  // Initialize a `thrust::device_vector` of length `n_vertices` for k-core
  // values
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<int> k_cores(n_vertices);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::kcore::run(G, k_cores.data().get());

  // --
  // CPU Run

  thrust::host_vector<int> h_k_cores(n_vertices);

  float cpu_elapsed =
      kcore_cpu::run<csr_t, vertex_t, edge_t, weight_t>(csr, h_k_cores.data());

  int n_errors =
      util::compare(k_cores.data().get(), h_k_cores.data(), n_vertices);

  // --
  // Log + Validate

  print::head(k_cores, 40, "GPU k-core values");
  print::head(h_k_cores, 40, "CPU k-core values");

  // Print runtime returned by `gunrock::kcore::run`
  // This will just be the GPU runtime of the "region of interest", and will
  // ignore any setup/teardown code.
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_kcore(argc, argv);
}
