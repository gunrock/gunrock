#include <nvbench/nvbench.cuh>
#include <cxxopts.hpp>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/geo.hxx>

using namespace gunrock;
using namespace memory;

using vertex_t = int;
using edge_t = int;
using weight_t = float;

std::string matrix_filename;
std::string coordinates_filename;

struct parameters_t {
  std::string matrix_filename;
  std::string coordinates_filename;
  bool help = false;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv) : options(argv[0], "Geo Benchmarking") {
    options.allow_unrecognised_options();
    // Add command line options
    options.add_options()("h,help", "Print help")  // help
        ("m,market", "Matrix file",
         cxxopts::value<std::string>())  // mtx
        ("c,coordinates", "Coordinates file",
         cxxopts::value<std::string>());  // coords

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      help = true;
      std::cout << options.help({""});
      std::cout << "  [optional nvbench args]" << std::endl << std::endl;
      // Do not exit so we also print NVBench help.
    } else {
      if (result.count("market") == 1) {
        matrix_filename = result["market"].as<std::string>();
        if (!util::is_market(matrix_filename)) {
          std::cout << options.help({""});
          std::cout << "  [optional nvbench args]" << std::endl << std::endl;
          std::exit(0);
        }
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
      if (result.count("coordinates") == 1) {
        coordinates_filename = result["coordinates"].as<std::string>();
      } else {
        std::cout << options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
    }
  }
};

/**
 * @brief Reads a coordinates file from an input-stream a dense array.
 *
 * Here is an example of the labels file
 * +-------------------------+
 * |%%Labels Formatted File  | <-- header
 * |%                        | <-+
 * |% comments               |   |-- comments
 * |%                        | <-+
 * |  N L L                  | <-- num_nodes, num_labels, num_labels
 * |  I0 L1A L1B             | <-- node id, latitude, longutude
 * |  I4                     | <-- coordinates missing, populated as invalids
 * |  I5 L5A L5B             |
 * |  . . .                  |
 * |  IN LNA LNB             |
 * +-------------------------+
 *
 * @note Node ID (first column) must be 0-based.
 * @note If a Node ID is present but coordinates are missing,
 *       the coordinates are filled as invalids.
 * @note If Node ID and coordinates are missing, the coordinates
 *       for those Node IDs are filled as invalids.
 */
void read_coordinates_file(std::string filename,
                           geo::coordinates_t* coordinates) {
  FILE* f_in = fopen(filename.c_str(), "r");
  int labels_read = gunrock::numeric_limits<int>::invalid();
  char line[1024];

  while (true) {
    if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
      break;
    }

    if (line[0] == '%') {  // Comment
      if (strlen(line) >= 2 && line[1] == '%') {
        // Header -> Can be used to extract info for labels
      }
    }  // -> if

    else if (!gunrock::util::limits::is_valid(
                 labels_read)) {  // Problem description-> First line
                                  // with nodes and labels info
      long long ll_nodes, ll_label_x, ll_label_y;
      int items_scanned =
          sscanf(line, "%lld %lld %lld", &ll_nodes, &ll_label_x, &ll_label_y);
      labels_read = 0;
    }  // -> else if

    else {                // Now we can start storing labels
      long long ll_node;  // Active node

      // Used for sscanf
      float lf_label_a = gunrock::numeric_limits<float>::invalid();
      float lf_label_b = gunrock::numeric_limits<float>::invalid();

      float ll_label_a, ll_label_b;  // Used to parse float/double

      int num_input =
          sscanf(line, "%lld %f %f", &ll_node, &lf_label_a, &lf_label_b);

      if (num_input == 1) {
        // if only node id exists in the line, populate the coordinates with
        // invalid values.
        ll_label_a = gunrock::numeric_limits<float>::invalid();
        ll_label_b = gunrock::numeric_limits<float>::invalid();
      }

      else if (num_input == 3) {
        // if all three; node id, latitude and longitude exist, populate all
        // three.
        ll_label_a = lf_label_a;
        ll_label_b = lf_label_b;

        labels_read++;
      }

      else {
        // else print an error.
        std::cerr << "Invalid coordinates file format." << std::endl;
        exit(1);
      }

      // XXX: Make sure these are 0-based;
      coordinates[ll_node].latitude = ll_label_a;
      coordinates[ll_node].longitude = ll_label_b;

    }  // -> else
  }    // -> while

  if (labels_read) {
    std::cout << "Valid coordinates read: " << labels_read << std::endl;
  } else if (labels_read <= 0) {
    std::cerr << "Error: No coordinates read." << std::endl;
    exit(1);
  }
}

void geo_bench(nvbench::state& state) {
  // --
  // Add metrics
  state.collect_dram_throughput();
  state.collect_l1_hit_rates();
  state.collect_l2_hit_rates();
  state.collect_loads_efficiency();
  state.collect_stores_efficiency();

  // --
  // Define types
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // Build graph + metadata
  csr_t csr;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  csr.from_coo(mm.load(matrix_filename));

  thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
  thrust::device_vector<vertex_t> column_indices(csr.number_of_nonzeros);
  thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);

  auto G = graph::build::from_csr<memory_space_t::device,
                                  graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get(),  // values
      row_indices.data().get(),         // row_indices
      column_offsets.data().get()       // column_offsets
  );

  // --
  // Params and memory allocation
  unsigned int spatial_iterations = 1000;
  unsigned int total_iterations = 10;

  vertex_t n_vertices = G.get_number_of_vertices();

  // Coordinates: Latitude/Longitude
  geo::coordinates_t default_invalid;
  default_invalid.latitude = gunrock::numeric_limits<float>::invalid();
  default_invalid.longitude = gunrock::numeric_limits<float>::invalid();

  thrust::host_vector<geo::coordinates_t> load_coordinates(n_vertices,
                                                           default_invalid);
  read_coordinates_file(coordinates_filename, load_coordinates.data());
  thrust::device_vector<geo::coordinates_t> coordinates(load_coordinates);

  // --
  // Run Geo with NVBench
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    gunrock::geo::run(G, coordinates.data().get(), total_iterations,
                      spatial_iterations);
  });
}

int main(int argc, char** argv) {
  parameters_t params(argc, argv);
  matrix_filename = params.matrix_filename;
  coordinates_filename = params.coordinates_filename;

  if (params.help) {
    // Print NVBench help.
    const char* args[1] = {"-h"};
    NVBENCH_MAIN_BODY(1, args);
  } else {
    // Create a new argument array without matrix and coordinate filenames to
    // pass to NVBench.
    char* args[argc - 4];
    int j = 0;
    for (int i = 0; i < argc; i++) {
      if (strcmp(argv[i], "--market") == 0 || strcmp(argv[i], "-m") == 0 ||
          strcmp(argv[i], "--coordinates") == 0 || strcmp(argv[i], "-c") == 0) {
        i++;
        continue;
      }
      args[j] = argv[i];
      j++;
    }

    NVBENCH_BENCH(geo_bench);
    NVBENCH_MAIN_BODY(argc - 4, args);
  }
}
