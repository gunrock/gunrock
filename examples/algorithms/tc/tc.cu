#include <gunrock/algorithms/tc.hxx>

using namespace gunrock;
using namespace memory;

void test_tc(int num_arguments, char** argument_array) {
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/tc filename.mtx reduce" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = uint32_t;
  using edge_t = uint32_t;
  using weight_t = float;
  using count_t = vertex_t;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;

  // --
  // IO

  const std::string filename = argument_array[1];
  const std::string reduce = argument_array[2];
  const bool reduce_all_triangles = reduce.find("true") != std::string::npos;

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto mmatrix = mm.load(filename);
    if (!mm_is_symmetric(mm.code)) {
      std::cerr << "Error: input matrix must be symmetric" << std::endl;
      exit(1);
    }
    csr.from_coo(mmatrix);
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device,
                                  graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<count_t> triangles_count(n_vertices, 0);

  // --
  // GPU Run

  std::size_t total_triangles = 0;
  float gpu_elapsed = tc::run(G, reduce_all_triangles,
                              triangles_count.data().get(), &total_triangles);

  // --
  // Log

  print::head(triangles_count, 40, "Per-vertex triangle count");
  if (reduce_all_triangles) {
    std::cout << "Total Graph Traingles : " << total_triangles << std::endl;
  }
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_tc(argc, argv);
}
