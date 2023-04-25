#include <set>

#include <gunrock/algorithms/color.hxx>
#include "color_cpu.hxx"  // Reference implementation

using namespace gunrock;
using namespace memory;

void test_color(int num_arguments, char** argument_array) {
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

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  csr_t csr;

  csr.from_coo(coo);

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> colors(n_vertices);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::color::run(G, colors.data().get());

  // --
  // CPU Run

  thrust::host_vector<vertex_t> h_colors(n_vertices);

  float cpu_elapsed =
      color_cpu::run<csr_t, vertex_t, edge_t, weight_t>(csr, h_colors.data());

  int n_errors = color_cpu::compute_error<csr_t, vertex_t, edge_t, weight_t>(
      csr, colors, h_colors);

  std::vector<int> stl_colors(n_vertices);
  thrust::copy(colors.begin(), colors.end(), stl_colors.begin());
  int n_colors = std::set(stl_colors.begin(), stl_colors.end()).size();

  // --
  // Log
  print::head(colors, 40, "GPU colors");
  print::head(h_colors, 40, "CPU colors");

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
  std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  std::cout << "Number of colors : " << n_colors << std::endl;
  std::cout << "Number of errors : " << n_errors << std::endl;
}

int main(int argc, char** argv) {
  test_color(argc, argv);
}
