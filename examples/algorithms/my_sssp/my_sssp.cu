#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/my_sssp.hxx>  // Won't compile yet

using namespace gunrock;
using namespace memory;

void test_my_sssp(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(filename);

  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(coo);

  auto G = graph::build(properties, csr);

  std::cout << "G.get_number_of_vertices() : " << G.get_number_of_vertices() << std::endl;
  std::cout << "G.get_number_of_edges()    : " << G.get_number_of_edges()    << std::endl;

  // Params
  vertex_t single_source = 0;
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> distances(n_vertices);

  // Run
  float gpu_elapsed = gunrock::my_sssp::run(G, single_source, distances.data().get());

  // Output
  std::cout << "GPU Distances (output) = ";
  thrust::copy(distances.begin(), distances.end(), std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_my_sssp(argc, argv);
}