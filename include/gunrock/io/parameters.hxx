#include <cxxopts.hpp>

namespace gunrock {
namespace io {
namespace cli {

struct parameters_t {
  std::string filename;
  int source = -1;
  std::string json_dir = ".";
  std::string json_file = "";
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
        ("n,num_runs", "Number of runs", cxxopts::value<int>())     // runs
        ("d,json_dir", "JSON output directory",
         cxxopts::value<std::string>())  // json output directory
        ("f,json_file", "JSON output file",
         cxxopts::value<std::string>());  // json output file

    if (algorithm == "bc" || algorithm == "bfs" || algorithm == "sssp") {
      options.add_options()("s,source", "Starting source (random if omitted)",
                            cxxopts::value<int>());  // source
      if (algorithm == "bfs" || algorithm == "sssp") {
        options.add_options()("validate", "CPU validation");  // validate
      }
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

    if (result.count("source") == 1) {
      source = result["source"].as<int>();
    }

    if (result.count("json_dir") == 1) {
      json_dir = result["json_dir"].as<std::string>();
    }

    if (result.count("json_file") == 1) {
      json_file = result["json_file"].as<std::string>();
    }
  }
};

}  // namespace cli
}  // namespace io
}  // namespace gunrock