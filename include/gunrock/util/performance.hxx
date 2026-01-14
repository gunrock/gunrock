#ifdef _WIN32
#include <gunrock/util/sysinfo.hxx>
#else
#include <sys/utsname.h>
#endif

#include "nlohmann/json.hpp"
#include <time.h>
#include <regex>
#include <filesystem>
#include <fstream>
#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/util/compiler.hxx>
#include <gunrock/io/git.hxx>

namespace gunrock {
namespace util {
namespace stats {

using vertex_t = int;
using edge_t = int;

// Date JSON schema was last updated
std::string schema_version = "2022-10-28";

class system_info_t {
 private:
  struct utsname uts;

 public:
  system_info_t() { uname(&uts); }

  std::string sysname() const { return std::string(uts.sysname); }
  std::string release() const { return std::string(uts.release); }
  std::string version() const { return std::string(uts.version); }
  std::string machine() const { return std::string(uts.machine); }
  std::string nodename() const { return std::string(uts.nodename); }

  void get_sys_info(nlohmann::json* jsn) {
    std::map<std::string, std::string> system_info;
    system_info["sysname"] = sysname();
    system_info["release"] = release();
    system_info["version"] = version();
    system_info["machine"] = machine();
    system_info["nodename"] = nodename();
    jsn->push_back(
        nlohmann::json::object_t::value_type("sysinfo", system_info));
  }
};

void get_gpu_info(nlohmann::json* jsn) {
  gunrock::gcuda::device_properties_t device_properties;
  std::map<std::string, std::string> gpuinfo;

  // If no valid devices
  if (gunrock::gcuda::properties::device_count() == 0) {
    return;
  }

  gunrock::gcuda::properties::set_device_properties(&device_properties);
  gpuinfo["name"] = gunrock::gcuda::properties::gpu_name(device_properties);
  gpuinfo["total_global_mem"] = std::to_string(
      gunrock::gcuda::properties::total_global_memory(device_properties));
  gpuinfo["major"] =
      std::to_string(gunrock::gcuda::properties::sm_major(device_properties));
  gpuinfo["minor"] =
      std::to_string(gunrock::gcuda::properties::sm_minor(device_properties));
  gpuinfo["clock_rate"] =
      std::to_string(gunrock::gcuda::properties::clock_rate(device_properties));
  gpuinfo["multi_processor_count"] = std::to_string(
      gunrock::gcuda::properties::multi_processor_count(device_properties));
  gpuinfo["driver_version"] =
      std::to_string(gunrock::gcuda::properties::driver_version());
  gpuinfo["runtime_version"] =
      std::to_string(gunrock::gcuda::properties::runtime_version());
  gpuinfo["compute_version"] = std::to_string(
      gunrock::gcuda::properties::compute_version(device_properties));

  jsn->push_back(nlohmann::json::object_t::value_type("gpuinfo", gpuinfo));
}

void export_performance_stats(
    std::vector<benchmark::host_benchmark_t>& benchmark_metrics,
    size_t edges,
    size_t vertices,
    std::vector<float>& run_times,
    std::string primitive,
    std::string filename,
    std::string graph_type,
    std::string json_dir,
    std::string json_file,
    std::vector<int>& sources,
    std::vector<std::string>& tags,
    int argc,
    char** argv) {
  float avg_run_time;
  float stdev_run_times;
  float min_run_time;
  float max_run_time;
  std::string time_s;
  nlohmann::json jsn;
  std::string json_dir_file;
  std::string command_line_call;

  if (run_times.size() == 1) {
    avg_run_time = run_times[0];
    stdev_run_times = 0;
    min_run_time = avg_run_time;
    max_run_time = avg_run_time;
  } else {
    // Get average run time
    avg_run_time = std::accumulate(run_times.begin(), run_times.end(), 0.0) /
                   run_times.size();

    // Get run time standard deviation
    std::vector<double> diff(run_times.size());
    std::transform(run_times.begin(), run_times.end(), diff.begin(),
                   [avg_run_time](double x) { return x - avg_run_time; });
    double sq_sum =
        std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    stdev_run_times = std::sqrt(sq_sum / run_times.size());

    // Get min and max run times
    min_run_time = *std::min_element(run_times.begin(), run_times.end());
    max_run_time = *std::max_element(run_times.begin(), run_times.end());
  }
  // Get time info
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, sizeof(buffer), "%a %b %d %H:%M:%S %Y", timeinfo);
  time_s = buffer;

  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  std::string time_ms = std::to_string(ms.count());

  // Get command line call
  for (int i = 0; i < argc; i++) {
    command_line_call += argv[i];
    command_line_call += " ";
  }
  command_line_call = command_line_call.substr(0, command_line_call.size() - 1);

  // Write values to JSON object
  jsn.push_back(nlohmann::json::object_t::value_type("engine", "Essentials"));
  jsn.push_back(nlohmann::json::object_t::value_type("primitive", primitive));
  jsn.push_back(nlohmann::json::object_t::value_type("graph_type", graph_type));
  jsn.push_back(nlohmann::json::object_t::value_type("num_edges", edges));
  jsn.push_back(nlohmann::json::object_t::value_type("num_vertices", vertices));
  jsn.push_back(nlohmann::json::object_t::value_type("srcs", sources));
  jsn.push_back(nlohmann::json::object_t::value_type("tags", tags));
  jsn.push_back(
      nlohmann::json::object_t::value_type("process_times", run_times));
  jsn.push_back(nlohmann::json::object_t::value_type("graph_file", filename));
  jsn.push_back(
      nlohmann::json::object_t::value_type("avg_process_time", avg_run_time));
  jsn.push_back(nlohmann::json::object_t::value_type("stddev_process_time",
                                                     stdev_run_times));
  jsn.push_back(
      nlohmann::json::object_t::value_type("min_process_time", min_run_time));
  jsn.push_back(
      nlohmann::json::object_t::value_type("max_process_time", max_run_time));
  jsn.push_back(nlohmann::json::object_t::value_type("time", time_s));
  jsn.push_back(
      nlohmann::json::object_t::value_type("command_line", command_line_call));
  jsn.push_back(nlohmann::json::object_t::value_type(
      "git_commit_sha", gunrock::io::git_commit_sha1()));
  jsn.push_back(nlohmann::json::object_t::value_type(
      "schema", gunrock::util::stats::schema_version));
  jsn.push_back(nlohmann::json::object_t::value_type(
      "compiler", gunrock::util::stats::compiler));
  jsn.push_back(nlohmann::json::object_t::value_type(
      "compiler_version", gunrock::util::stats::compiler_version));

  // Include additional stats if this is a full performance run
  std::vector<int> search_depths;
  std::vector<unsigned int> nodes_visited;
  std::vector<unsigned int> edges_visited;

#if ESSENTIALS_COLLECT_METRICS
  int avg_search_depth;
  int min_search_depth;
  int max_search_depth;
  float avg_mteps;
  float min_mteps;
  float max_mteps;

  std::transform(benchmark_metrics.begin(), benchmark_metrics.end(),
                 std::back_inserter(search_depths),
                 [](benchmark::host_benchmark_t const& b) -> int {
                   return b.search_depth;
                 });

  std::transform(benchmark_metrics.begin(), benchmark_metrics.end(),
                 std::back_inserter(nodes_visited),
                 [](benchmark::host_benchmark_t const& b) -> unsigned int {
                   return b.vertices_visited;
                 });

  std::transform(benchmark_metrics.begin(), benchmark_metrics.end(),
                 std::back_inserter(edges_visited),
                 [](benchmark::host_benchmark_t const& b) -> unsigned int {
                   return b.edges_visited;
                 });

  // Get average search depth
  avg_search_depth =
      std::reduce(search_depths.begin(), search_depths.end(), 0.0) /
      search_depths.size();

  // Get min and max search depths
  min_search_depth =
      *std::min_element(search_depths.begin(), search_depths.end());
  max_search_depth =
      *std::max_element(search_depths.begin(), search_depths.end());

  // Get MTEPS
  std::vector<float> mteps(edges_visited.size());
  std::transform(edges_visited.begin(), edges_visited.end(), run_times.begin(),
                 mteps.begin(), std::divides<float>());
  std::transform(mteps.begin(), mteps.end(), mteps.begin(),
                 [](auto& c) { return c / 1000; });

  avg_mteps = std::reduce(mteps.begin(), mteps.end(), 0.0) / mteps.size();

  min_mteps = *std::min_element(mteps.begin(), mteps.end());
  max_mteps = *std::max_element(mteps.begin(), mteps.end());

  jsn.push_back(
      nlohmann::json::object_t::value_type("edges_visited", edges_visited));
  jsn.push_back(
      nlohmann::json::object_t::value_type("nodes_visited", nodes_visited));
  jsn.push_back(
      nlohmann::json::object_t::value_type("search_depths", search_depths));
  jsn.push_back(nlohmann::json::object_t::value_type("avg_search_depth",
                                                     avg_search_depth));
  jsn.push_back(nlohmann::json::object_t::value_type("min_search_depth",
                                                     min_search_depth));
  jsn.push_back(nlohmann::json::object_t::value_type("max_search_depth",
                                                     max_search_depth));
  jsn.push_back(nlohmann::json::object_t::value_type("mteps", mteps));
  jsn.push_back(nlohmann::json::object_t::value_type("avg_mteps", avg_mteps));
  jsn.push_back(nlohmann::json::object_t::value_type("min_mteps", min_mteps));
  jsn.push_back(nlohmann::json::object_t::value_type("max_mteps", max_mteps));
#endif

  // Get GPU info
  get_gpu_info(&jsn);

  // Get System info
  system_info_t sys;
  sys.get_sys_info(&jsn);

  // Write JSON to file
  if (json_file == "") {
    std::string time_str_filename = time_s.substr(0, time_s.size() - 4) +
                                    time_ms + '_' +
                                    time_s.substr(time_s.length() - 4);
    std::string fn =
        std::filesystem::path(filename).filename().generic_string();
    int last = fn.find_last_of(".");
    fn = fn.substr(0, last);
    time_str_filename =
        std::regex_replace(time_str_filename, std::regex(" "), "_");
    time_str_filename =
        std::regex_replace(time_str_filename, std::regex(":"), "");
    json_dir_file = json_dir + "/" + primitive + "_" + fn + "_" +
                    time_str_filename + ".json";
  } else {
    json_dir_file = json_dir + "/" + json_file;
  }

  std::ofstream outfile(json_dir_file);
  outfile << jsn.dump(4);
  outfile.close();
}

}  // namespace stats
}  // namespace util
}  // namespace gunrock