#include <cxxopts.hpp>

#include <gunrock/error.hxx>                        // error checking
#include <gunrock/graph/graph.hxx>                  // graph class
#include <gunrock/formats/formats.hxx>              // csr support
#include <gunrock/cuda/cuda.hxx>                    // context to run on
#include <gunrock/framework/operators/for/for.hxx>  // for operator

void sample_cxxopts(int argc, char** argv) {
  using namespace gunrock;

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  cxxopts::Options options(argv[0], "Gunrock for operator test");
  options.add_options()  // Allows to add options.
      ("c,csr", "CSR binary file",
       cxxopts::value<std::string>())  // CSR
      ("m,market", "Matrix-market format file",
       cxxopts::value<std::string>())  // Market
      ("d,device", "Device to run on",
       cxxopts::value<int>()->default_value("0"))  // Device
      ("v,verbose", "Verbose output",
       cxxopts::value<bool>()->default_value("false"))  // Verbose (not used)
      ("h,help", "Print help");                         // Help

  auto result = options.parse(argc, argv);

  if (result.count("help") ||
      (result.count("market") == 0 && result.count("csr") == 0)) {
    std::cout << options.help({""}) << std::endl;
    exit(0);
  }

  csr_t csr;

  if (result.count("market") == 1) {
    std::string filename = result["market"].as<std::string>();
    if (util::is_market(filename)) {
      io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
      csr.from_coo(mm.load(filename));
    } else {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
  } else if (result.count("csr") == 1) {
    std::string filename = result["csr"].as<std::string>();
    if (util::is_binary_csr(filename))
      csr.read_binary(filename);
    else {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
  } else {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
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
  );

  // Initialize the context.
  cuda::device_id_t device = result["device"].as<int>();
  cuda::multi_context_t context(device);

  auto vertex_op = [=] __device__(vertex_t const& v) -> void {
    printf("%i\n", v);
  };

  operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
      G,          // graph
      vertex_op,  // lambda function
      context     // context
  );
}