#include <set>

#include <gunrock/applications/color.hxx>

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

  // --
  // IO

  std::string filename = argument_array[1];

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.from_coo(mm.load(filename));

  // --
  // Build graph + metadata

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
  thrust::device_vector<vertex_t> colors(n_vertices);

  // --
  // Run problem

  float elapsed = gunrock::color::run(G, colors.data().get());

  // --
  // Log

  std::cout << "Colors (output) = ";
  thrust::copy(colors.begin(), colors.end(),
               std::ostream_iterator<weight_t>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "color Elapsed Time: " << elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_color(argc, argv);
}
