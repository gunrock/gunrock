#include <gunrock/algorithms/spmv.hxx>
#include <gunrock/algorithms/generate/random.hxx>
#include "spmv_cpu.hxx"

using namespace gunrock;
using namespace memory;

void test_spmv(int num_arguments, char** argument_array) {
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
  // See `format` to see other supported formats.
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

  // --
  // Params and memory allocation
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> x(n_vertices);
  thrust::device_vector<weight_t> y(n_vertices);

  gunrock::generate::random::uniform_distribution(x);

  // --
  // GPU Run
  float gpu_elapsed = gunrock::spmv::run(G, x.data().get(), y.data().get());

  // --
  // CPU Run

  thrust::host_vector<weight_t> y_h(n_vertices);
  float cpu_elapsed = spmv_cpu::run(csr, x, y_h);

  // --
  // Log + Validate
  int n_errors = util::compare(
      y.data().get(), y_h.data(), n_vertices,
      [=](const weight_t a, const weight_t b) {
        // TODO: needs better accuracy.
        return std::abs(a - b) > 1e-2;
      },
      true);

  gunrock::print::head(y, 40, "GPU y-vector");
  gunrock::print::head(y_h, 40, "CPU y-vector");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

// Main method, wrapping test function
int main(int argc, char** argv) {
  test_spmv(argc, argv);
}