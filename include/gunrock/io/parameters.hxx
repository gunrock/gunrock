#include <cxxopts.hpp>

namespace gunrock {
namespace io {
namespace cli {

struct parameters_t {
  std::string filename;
  std::string source_string = "";
  std::string json_dir = ".";
  std::string json_file = "";
  std::string tag_string = "";
  int num_runs = 1;
  cxxopts::Options options;
  bool collect_metrics = false;
  bool validate = false;
  bool binary = false;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv, std::string algorithm)
      : options(argv[0], algorithm + " example") {
    // Add command line options
    options.add_options()("help", "Print help")  // help
        ("collect_metrics",
         "collect performance analysis metrics")  // performance evaluation
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx file
        ("d,json_dir", "JSON output directory",
         cxxopts::value<std::string>())  // json output directory
        ("f,json_file", "JSON output file",
         cxxopts::value<std::string>())  // json output file
        ("t,tag", "Tags for the JSON output; comma-separated string of tags",
         cxxopts::value<std::string>());  // tags

    // Algorithms with sources
    if (algorithm == "Betweenness Centrality" ||
        algorithm == "Breadth First Search" ||
        algorithm == "Single Source Shortest Path") {
      options.add_options()("s,src",
                            "Source(s) (random if omitted); "
                            "comma-separated string of ints",
                            cxxopts::value<std::string>())  // source
          ("n,num_runs", "Number of runs (ignored if multiple sources passed)",
           cxxopts::value<int>());  // runs
      if (algorithm == "Breadth First Search" ||
          algorithm == "Single Source Shortest Path") {
        options.add_options()("validate", "CPU validation");  // validate
      }
    } else {
      options.add_options()("n,num_runs", "Number of runs",
                            cxxopts::value<int>());  // runs
    }

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") == 1) {
      filename = result["market"].as<std::string>();
      if (util::is_binary_csr(filename)) {
        binary = true;
      } else if (!util::is_market(filename)) {
        std::cout << options.help({""}) << std::endl;
        std::exit(0);
      }
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("validate") == 1) {
      validate = true;
    }

    if (result.count("collect_metrics") == 1) {
      collect_metrics = true;
    }

    if (result.count("num_runs") == 1) {
      num_runs = result["num_runs"].as<int>();
    }

    if (result.count("tag") == 1) {
      tag_string = result["tag"].as<std::string>();
    }

    if (result.count("src") == 1) {
      source_string = result["src"].as<std::string>();
    }

    if (result.count("json_dir") == 1) {
      json_dir = result["json_dir"].as<std::string>();
    }

    if (result.count("json_file") == 1) {
      json_file = result["json_file"].as<std::string>();
    }
  }
};

void parse_source_string(std::string source_str,
                         std::vector<int>* source_vect,
                         int n_vertices,
                         int n_runs) {
  if (source_str == "") {
    // Generate random starting source
    std::random_device seed;
    std::mt19937 engine(seed());
    for (int i = 0; i < n_runs; i++) {
      std::uniform_int_distribution<int> dist(0, n_vertices - 1);
      source_vect->push_back(dist(engine));
    }
  } else {
    std::stringstream ss(source_str);
    while (ss.good()) {
      std::string source;
      getline(ss, source, ',');
      int source_int;
      try {
        source_int = std::stoi(source);
      } catch (...) {
        std::cout << "Error: Invalid source"
                  << "\n";
        exit(1);
      }
      if (source_int >= 0 && source_int < n_vertices) {
        source_vect->push_back(source_int);
      } else {
        std::cout << "Error: Invalid source"
                  << "\n";
        exit(1);
      }
    }
    if (source_vect->size() == 1) {
      source_vect->insert(source_vect->end(), n_runs - 1, source_vect->at(0));
    }
  }
}

void parse_tag_string(std::string tag_str, std::vector<std::string>* tag_vect) {
  std::stringstream ss(tag_str);
  while (ss.good()) {
    std::string tag;
    getline(ss, tag, ',');
    if (tag != "") {
      tag_vect->push_back(tag);
    }
  }
}

}  // namespace cli
}  // namespace io
}  // namespace gunrock