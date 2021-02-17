#include <cstdlib>  // EXIT_SUCCESS
#include <iostream>
#include <sstream>

#include <gunrock/applications/geo.hxx>

using namespace gunrock;
using namespace memory;

class coordinate_adaptor {
 public:
  const geo::coordinates_t& m;
  coordinate_adaptor(const geo::coordinates_t& a) : m(a) {}

  friend std::ostream& operator<<(std::ostream& out,
                                  const coordinate_adaptor& d) {
    const geo::coordinates_t& m = d.m;
    return out << m.latitude << ", " << m.longitude;
  }
};

/**
 * @brief Reads a coordinates file from an input-stream a dense array.
 *
 * Here is an example of the labels format
 * +----------------------------------------------+
 * |%%Labels Formatted File 			  | <--- header line
 * |%                                             | <--+
 * |% comments                                    |    |-- 0 or more comment
 * lines
 * |%                                             | <--+
 * |  N L L                                       | <--- nodes, labels, labels
 * |  I1 L1A L1B                                  | <--+
 * |  I2 L2A L2B                                  |    |
 * |     . . .                                    |    |
 * |  IN LNA LNB                                  | <--+
 * +----------------------------------------------+
 */
void read_coordinates_file(
    std::string filename,
    thrust::device_vector<geo::coordinates_t>& _coordinates) {
  geo::coordinates_t default_invalid{gunrock::numeric_limits<float>::invalid(),
                                     gunrock::numeric_limits<float>::invalid()};
  thrust::host_vector<geo::coordinates_t> coordinates(_coordinates.size(),
                                                      default_invalid);
  FILE* f_in = fopen(filename.c_str(), "r");
  int labels_read = gunrock::numeric_limits<int>::invalid();
  long long nodes = 0;
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
      nodes = ll_nodes;

      for (int k = 0; k < nodes; k++) {
        coordinates[k].latitude = gunrock::numeric_limits<float>::invalid();
        coordinates[k].longitude = gunrock::numeric_limits<float>::invalid();
      }

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

      coordinates[ll_node - 1].latitude = ll_label_a;
      coordinates[ll_node - 1].longitude = ll_label_b;
    }  // -> else
  }    // -> while

  if (labels_read) {
    std::cout << "Valid coordinates read: " << labels_read << std::endl;
  } else if (labels_read <= 0) {
    std::cerr << "Error: No coordinates read." << std::endl;
    exit(1);
  }

  // Move to device.
  _coordinates = coordinates;
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
      format::csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t>;
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
  thrust::device_vector<geo::coordinates_t> coordinates(n_vertices);

  // Coordinates: Latitude/Longitude
  std::string coordinates_filename = argument_array[2];
  read_coordinates_file(coordinates_filename, coordinates);

  // --
  // GPU Run

  float gpu_elapsed = gunrock::geo::run(G, coordinates.data().get(),
                                        total_iterations, spatial_iterations);

  // --
  // Log + Validate

  std::cout << "Coordinates (output) = " << std::endl;

  thrust::host_vector<geo::coordinates_t> h_coordinates = coordinates;
  for (int i = 0; i < h_coordinates.size(); i++)
    std::cout << "Node (" << i << ") = " << h_coordinates[i].latitude << ", "
              << h_coordinates[i].longitude << std::endl;

  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_geo(argc, argv);
  return EXIT_SUCCESS;
}
