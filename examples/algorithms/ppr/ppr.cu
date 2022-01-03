#include <gunrock/algorithms/ppr.hxx>
#include "ppr_cpu.hxx"

using namespace gunrock;
using namespace memory;

void test_ppr(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;

  // --
  // IO

  weight_t alpha = 0.15;
  weight_t epsilon = 1e-6;
  vertex_t n_seeds = 10;

  std::string filename = argument_array[1];

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
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

  thrust::device_vector<weight_t> p(n_seeds * n_vertices);

  // --
  // GPU Run

  float gpu_elapsed =
      gunrock::ppr::run_batch(G, n_seeds, p.data().get(), alpha, epsilon);

  // --
  // CPU Run

  thrust::host_vector<weight_t> h_p(n_seeds * n_vertices);

  float cpu_elapsed = ppr_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
      csr, n_seeds, h_p.data(), alpha, epsilon);

  int n_errors = util::compare(p.data().get(), h_p.data(), n_seeds * n_vertices,
                               [epsilon](const weight_t a, const weight_t b) {
                                 return std::abs(a - b) > epsilon;
                               });

  // --
  // Log + Validate

  print::head(p, 40, "GPU rank");
  print::head(h_p, 40, "CPU rank");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

int main(int argc, char** argv) {
  test_ppr(argc, argv);
}
