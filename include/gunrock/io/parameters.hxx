#include <cxxopts.hpp>
#include <string>
#include <algorithm>
#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace io {
namespace cli {

// Forward declarations
operators::load_balance_t parse_load_balance(std::string str);
operators::filter_algorithm_t parse_filter_algorithm(std::string str);
operators::uniquify_algorithm_t parse_uniquify_algorithm(std::string str);

struct parameters_t {
  std::string filename;
  std::string source_string = "";
  std::string json_dir = ".";
  std::string json_file = "";
  std::string tag_string = "";
  int num_runs = 1;
  cxxopts::Options options;
  bool export_metrics = false;
  bool validate = false;
  bool binary = false;
  
  // Operator configuration parameters
  operators::load_balance_t advance_load_balance = operators::load_balance_t::block_mapped;
  operators::filter_algorithm_t filter_algorithm = operators::filter_algorithm_t::predicated;
  bool enable_filter = false;
  
  // Uniquify operator configuration
  bool enable_uniquify = false;
  operators::uniquify_algorithm_t uniquify_algorithm = operators::uniquify_algorithm_t::unique;
  bool best_effort_uniquify = true;
  float uniquify_percent = 100.0f;

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
        ("export_metrics",
         "export performance analysis metrics")  // performance evaluation
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx file
        ("d,json_dir", "JSON output directory",
         cxxopts::value<std::string>())  // json output directory
        ("f,json_file", "JSON output file",
         cxxopts::value<std::string>())  // json output file
        ("t,tag", "Tags for the JSON output; comma-separated string of tags",
         cxxopts::value<std::string>())  // tags
        ("advance_load_balance", "Load balancing technique for advance operator (thread_mapped, block_mapped, merge_path, etc.)",
         cxxopts::value<std::string>())  // advance load balance
        ("filter_algorithm", "Filter algorithm (remove, predicated, compact, bypass)",
         cxxopts::value<std::string>())  // filter algorithm
        ("enable_filter", "Enable filter operator")  // enable filter
        ("enable_uniquify", "Enable uniquify operator")  // enable uniquify
        ("uniquify_algorithm", "Uniquify algorithm (unique, unique_copy)",
         cxxopts::value<std::string>())  // uniquify algorithm
        ("best_effort_uniquify", "Best-effort uniquification (skip sorting)")  // best effort
        ("uniquify_percent", "Percentage of elements to uniquify (0-100)",
         cxxopts::value<float>());  // uniquify percent

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

    if (result.count("export_metrics") == 1) {
      export_metrics = true;
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
    
    if (result.count("advance_load_balance") == 1) {
      advance_load_balance = parse_load_balance(result["advance_load_balance"].as<std::string>());
    }
    
    if (result.count("filter_algorithm") == 1) {
      filter_algorithm = parse_filter_algorithm(result["filter_algorithm"].as<std::string>());
    }
    
    if (result.count("enable_filter") == 1) {
      enable_filter = true;
    }
    
    if (result.count("enable_uniquify") == 1) {
      enable_uniquify = true;
    }
    
    if (result.count("uniquify_algorithm") == 1) {
      uniquify_algorithm = parse_uniquify_algorithm(result["uniquify_algorithm"].as<std::string>());
    }
    
    if (result.count("best_effort_uniquify") == 1) {
      best_effort_uniquify = true;
    }
    
    if (result.count("uniquify_percent") == 1) {
      uniquify_percent = result["uniquify_percent"].as<float>();
    }
  }
  
  /**
   * @brief Create an options_t struct from the parsed CLI arguments.
   * 
   * This helper method converts the CLI parameters into a gunrock::options_t
   * struct that can be passed to algorithm param_t constructors.
   * 
   * @return gunrock::options_t Options struct with CLI values.
   */
  gunrock::options_t get_options() const {
    gunrock::options_t opts;
    opts.advance_load_balance = advance_load_balance;
    opts.filter_algorithm = filter_algorithm;
    opts.enable_filter = enable_filter;
    opts.enable_uniquify = enable_uniquify;
    opts.uniquify_algorithm = uniquify_algorithm;
    opts.best_effort_uniquify = best_effort_uniquify;
    opts.uniquify_percent = uniquify_percent;
    return opts;
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

/**
 * @brief Parse load_balance_t enum from string.
 * 
 * @param str String representation (case-insensitive).
 * @return operators::load_balance_t Enum value.
 */
operators::load_balance_t parse_load_balance(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  
  if (str == "thread_mapped") return operators::load_balance_t::thread_mapped;
  if (str == "warp_mapped") return operators::load_balance_t::warp_mapped;
  if (str == "block_mapped") return operators::load_balance_t::block_mapped;
  if (str == "bucketing") return operators::load_balance_t::bucketing;
  if (str == "merge_path") return operators::load_balance_t::merge_path;
  if (str == "merge_path_v2") return operators::load_balance_t::merge_path_v2;
  if (str == "work_stealing") return operators::load_balance_t::work_stealing;
  
  // Default to block_mapped
  return operators::load_balance_t::block_mapped;
}

/**
 * @brief Parse filter_algorithm_t enum from string.
 * 
 * @param str String representation (case-insensitive).
 * @return operators::filter_algorithm_t Enum value.
 */
operators::filter_algorithm_t parse_filter_algorithm(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  
  if (str == "remove") return operators::filter_algorithm_t::remove;
  if (str == "predicated") return operators::filter_algorithm_t::predicated;
  if (str == "compact") return operators::filter_algorithm_t::compact;
  if (str == "bypass") return operators::filter_algorithm_t::bypass;
  
  // Default to predicated
  return operators::filter_algorithm_t::predicated;
}

/**
 * @brief Parse uniquify_algorithm_t enum from string.
 * 
 * @param str String representation (case-insensitive).
 * @return operators::uniquify_algorithm_t Enum value.
 */
operators::uniquify_algorithm_t parse_uniquify_algorithm(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  
  if (str == "unique") return operators::uniquify_algorithm_t::unique;
  if (str == "unique_copy") return operators::uniquify_algorithm_t::unique_copy;
  
  // Default to unique
  return operators::uniquify_algorithm_t::unique;
}

}  // namespace cli
}  // namespace io
}  // namespace gunrock