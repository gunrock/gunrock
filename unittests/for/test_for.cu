#include <cstdlib>  // EXIT_SUCCESS

#include <gunrock/error.hxx>                        // error checking
#include <gunrock/graph/graph.hxx>                  // graph class
#include <gunrock/formats/formats.hxx>              // csr support
#include <gunrock/cuda/cuda.hxx>                    // context to run on
#include <gunrock/framework/operators/for/for.hxx>  // for operator

void test_for(int num_arguments, char** argument_array) {
  using namespace gunrock;

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
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

  // Initialize the context.
  cuda::device_id_t device = 0;
  cuda::multi_context_t context(device);

  auto vertex_op = [=] __device__(vertex_t const& v) -> void {
    printf("%i\n", v);
  };

  auto edge_op = [=] __device__(edge_t const& e) -> void { printf("%i\n", e); };

  auto weight_op = [=] __device__(weight_t const& w) -> void {
    printf("%f\n", w);
  };

  operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
      G,          // graph
      vertex_op,  // lambda function
      context     // context
  );

  operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
      G,        // graph
      edge_op,  // lambda function
      context   // context
  );

  operators::parallel_for::execute<operators::parallel_for_each_t::weight>(
      G,          // graph
      weight_op,  // lambda function
      context     // context
  );
}

int main(int argc, char** argv) {
  test_for(argc, argv);
  return EXIT_SUCCESS;
}