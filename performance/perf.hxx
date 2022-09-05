#include <sys/utsname.h>
#include "nlohmann/json.hpp"
#include <string>
#include <gunrock/algorithms/algorithms.hxx>

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
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) /* no valid devices */
  {
    return;
  }
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&devProps, dev);

  int runtimeVersion, driverVersion;
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

void get_system_info(std::vector<std::vector<std::string>>* jsn) {}

void get_performance_stats(nlohmann::json* jsn,
                           int edges_visited,
                           int search_depth,
                           int gpu_elapsed) {
  get_gpu_info(jsn);
  Sysinfo sys;
  sys.get_sys_info(jsn);
  std::cout << *jsn << "\n";
}
