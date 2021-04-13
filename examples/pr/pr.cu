#include <gunrock/applications/pr.hxx>

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
  
  srand(time(NULL));
  
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<weight_t> p(n_vertices);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::pr::run(G, p.data().get());

  // --
  // Log + Validate

  thrust::host_vector<weight_t> p_h(p);

  std::cout << p_h[0] << std::endl; // PageRank: 0.003473
  std::cout << p_h[159452] << std::endl; // PageRank: 0.001432
  std::cout << p_h[78517] << std::endl; // PageRank: 0.001427
  std::cout << p_h[133417] << std::endl; // PageRank: 0.001421
  std::cout << p_h[144324] << std::endl; // PageRank: 0.001417
  std::cout << p_h[158200] << std::endl; // PageRank: 0.001416
  std::cout << p_h[2098] << std::endl; // PageRank: 0.001414
  std::cout << p_h[20982] << std::endl; // PageRank: 0.001413
  std::cout << p_h[115678] << std::endl; // PageRank: 0.001412
  std::cout << p_h[143635] << std::endl; // PageRank: 0.001398

  std::cout << "GPU p (output) = ";
  thrust::copy(p.begin(),
               (p.size() < 40) ? p.begin() + p.size()
                                       : p.begin() + 40,
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
