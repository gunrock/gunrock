#include <sys/utsname.h>
#include "nlohmann/json.hpp"
#include <time.h>
#include <regex>
#include <filesystem>
#include <fstream>

using vertex_t = int;
using edge_t = int;

class Sysinfo {
 private:
  struct utsname uts;

 public:
  Sysinfo() { uname(&uts); }

  std::string sysname() const { return std::string(uts.sysname); }
  std::string release() const { return std::string(uts.release); }
  std::string version() const { return std::string(uts.version); }
  std::string machine() const { return std::string(uts.machine); }
  std::string nodename() const { return std::string(uts.nodename); }

  void get_sys_info(nlohmann::json* jsn) {
    std::map<std::string, std::string> sysinfo;
    sysinfo["sysname"] = sysname();
    sysinfo["release"] = release();
    sysinfo["version"] = version();
    sysinfo["machine"] = machine();
    sysinfo["nodename"] = nodename();
    jsn->push_back(nlohmann::json::object_t::value_type("sysinfo", sysinfo));
  }
};

void get_gpu_info(nlohmann::json* jsn) {
  cudaDeviceProp devProps;
  int deviceCount;
  int dev = 0;
  int runtimeVersion, driverVersion;

  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) /* no valid devices */
  {
    return;
  }

  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&devProps, dev);

  cudaRuntimeGetVersion(&runtimeVersion);
  cudaDriverGetVersion(&driverVersion);

  std::map<std::string, std::string> gpuinfo;
  gpuinfo["name"] = devProps.name;
  gpuinfo["total_global_mem"] = std::to_string(devProps.totalGlobalMem);
  gpuinfo["major"] = std::to_string(devProps.major);
  gpuinfo["minor"] = std::to_string(devProps.minor);
  gpuinfo["clock_rate"] = std::to_string(devProps.clockRate);
  gpuinfo["multi_processor_count"] =
      std::to_string(devProps.multiProcessorCount);
  gpuinfo["driver_api"] = std::to_string(CUDA_VERSION);
  gpuinfo["driver_version"] = driverVersion;
  gpuinfo["runtime_version"] = runtimeVersion;
  gpuinfo["compute_version"] =
      std::to_string(devProps.major * 10 + devProps.minor);

  jsn->push_back(nlohmann::json::object_t::value_type("gpuinfo", gpuinfo));
}

void get_performance_stats(int edges_visited,
                           int nodes_visited,
                           edge_t edges,
                           vertex_t vertices,
                           int search_depth,
                           std::vector<float> run_times,
                           std::string primitive,
                           std::string filename,
                           std::string graph_type,
                           std::string json_dir,
                           std::string json_file) {
  float avg_run_time;
  float stdev_run_times;
  float min_run_time;
  float max_run_time;
  float avg_mteps;
  float min_mteps;
  float max_mteps;
  std::string time_s;
  nlohmann::json jsn;
  std::string json_dir_file;

  // Get average run time
  avg_run_time =
      std::reduce(run_times.begin(), run_times.end(), 0.0) / run_times.size();

  // Get run time standard deviation
  float sq_sum = std::inner_product(run_times.begin(), run_times.end(),
                                    run_times.begin(), 0.0);
  stdev_run_times =
      std::sqrt(sq_sum / run_times.size() - avg_run_time * avg_run_time);

  // Get min and max run times
  min_run_time = *std::min_element(run_times.begin(), run_times.end());
  max_run_time = *std::max_element(run_times.begin(), run_times.end());

  // Get MTEPS
  avg_mteps = (edges_visited / 1000) / avg_run_time;
  min_mteps = (edges_visited / 1000) / max_run_time;
  max_mteps = (edges_visited / 1000) / min_run_time;

  // Get time info
  time_t now = time(NULL);
  long ms;   // Milliseconds
  time_t s;  // Seconds
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);
  s = spec.tv_sec;
  ms = round(spec.tv_nsec / 1.0e6);  // Convert nanoseconds to milliseconds
  if (ms > 999) {
    s++;
    ms = 0;
  }
  time_s = std::string(ctime(&now));
  time_s = time_s.substr(0, time_s.size() - 1);
  std::string time_ms = std::to_string(ms);

  // Write values to JSON object
  jsn.push_back(nlohmann::json::object_t::value_type("engine", "Essentials"));
  jsn.push_back(nlohmann::json::object_t::value_type("primitive", primitive));
  jsn.push_back(nlohmann::json::object_t::value_type("graph-type", graph_type));
  jsn.push_back(
      nlohmann::json::object_t::value_type("edges-visited", edges_visited));
  jsn.push_back(
      nlohmann::json::object_t::value_type("nodes-visited", nodes_visited));
  jsn.push_back(nlohmann::json::object_t::value_type("num-edges", edges));
  jsn.push_back(nlohmann::json::object_t::value_type("num-vertices", vertices));
  jsn.push_back(
      nlohmann::json::object_t::value_type("process-times", run_times));
  jsn.push_back(
      nlohmann::json::object_t::value_type("search-depth", search_depth));
  jsn.push_back(nlohmann::json::object_t::value_type("graph-file", filename));
  jsn.push_back(
      nlohmann::json::object_t::value_type("avg-process-time", avg_run_time));
  jsn.push_back(nlohmann::json::object_t::value_type("stddev-process-time",
                                                     stdev_run_times));
  jsn.push_back(
      nlohmann::json::object_t::value_type("min-process-time", min_run_time));
  jsn.push_back(
      nlohmann::json::object_t::value_type("max-process-time", max_run_time));
  jsn.push_back(nlohmann::json::object_t::value_type("avg-mteps", avg_mteps));
  jsn.push_back(nlohmann::json::object_t::value_type("min-mteps", min_mteps));
  jsn.push_back(nlohmann::json::object_t::value_type("max-mteps", max_mteps));
  jsn.push_back(nlohmann::json::object_t::value_type("time", time_s));

  // Get GPU info
  get_gpu_info(&jsn);

  // Get System info
  Sysinfo sys;
  sys.get_sys_info(&jsn);

  // Write JSON to file
  if (json_file == "") {
    std::string time_str_filename = time_s.substr(0, time_s.size() - 4) +
                                    time_ms + '_' +
                                    time_s.substr(time_s.length() - 4);
    std::string fn = std::filesystem::path(filename).filename();
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
