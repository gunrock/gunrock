#include <iostream>
#include <sstream>

#include <gunrock/algorithms/geo.hxx>

using namespace gunrock;
using namespace memory;

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

void test_geo(int num_arguments, char** argument_array) {
  if (num_arguments != 3) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx filename.labels"
              << std::endl;
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

  unsigned int spatial_iterations = 1000;
  unsigned int total_iterations = 10;

  vertex_t n_vertices = G.get_number_of_vertices();

  // Coordinates: Latitude/Longitude
  std::string coordinates_filename = argument_array[2];
  geo::coordinates_t default_invalid;
  default_invalid.latitude = gunrock::numeric_limits<float>::invalid();
  default_invalid.longitude = gunrock::numeric_limits<float>::invalid();

  thrust::host_vector<geo::coordinates_t> load_coordinates(n_vertices,
                                                           default_invalid);
  read_coordinates_file(coordinates_filename, load_coordinates.data());
  thrust::device_vector<geo::coordinates_t> coordinates(load_coordinates);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::geo::run(G, coordinates.data().get(),
                                        total_iterations, spatial_iterations);

  // --
  // Log + Validate

  std::cout << "Coordinates (output) = " << std::endl;

  thrust::host_vector<geo::coordinates_t> h_coordinates = coordinates;
  auto h_coordinates_data = h_coordinates.data();
  for (int i = 0; i < h_coordinates.size() && i < 40; i++)
    std::cout << "Node (" << i << ") = " << h_coordinates_data[i].latitude
              << ", " << h_coordinates_data[i].longitude << std::endl;

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_geo(argc, argv);
}
