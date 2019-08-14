// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sysinfo.h
 *
 * @brief Reports information about your system (CPU/OS, GPU)
 */

#pragma once

#include <sys/utsname.h>      /* for Cpuinfo */
#include <cuda.h>             /* for Gpuinfo */
#include <cuda_runtime_api.h> /* for Gpuinfo */
#include <pwd.h>              /* for Userinfo */
#include <vector>
#include <cstring>

#include <unistd.h>
#include <sys/types.h>

namespace gunrock {
namespace util {

using namespace std;

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

  std::vector<std::pair<string, string>> getSysinfo() const {
    std::vector<std::pair<string, string>> json_sysinfo;
    json_sysinfo.push_back(make_pair("sysname", sysname()));
    json_sysinfo.push_back(make_pair("release", release()));
    json_sysinfo.push_back(make_pair("version", version()));
    json_sysinfo.push_back(make_pair("machine", machine()));
    json_sysinfo.push_back(make_pair("nodename", nodename()));

    return json_sysinfo;
  }
};

template <typename InfoT>
class Gpuinfo {
 public:
  /* TODO: Support  different pair types (other than string) */
  // std::vector<std::pair<string, string>> getGpuinfo(util::Info info) const {
  //   std::vector<std::pair<string, string>> info;
  //   cudaDeviceProp devProps;

  //   int deviceCount;
  //   cudaGetDeviceCount(&deviceCount);
  //   if (deviceCount == 0) /* no valid devices */
  //   {
  //     return info; /* empty */
  //   }
  //   int dev = 0;
  //   cudaGetDevice(&dev);
  //   cudaGetDeviceProperties(&devProps, dev);
  //   info.push_back(make_pair("name", devProps.name));
  //   info.push_back(make_pair("total_global_mem", std::to_string(int64_t(devProps.totalGlobalMem))));
  //   info.push_back(make_pair("major", std::to_string(devProps.major)));
  //   info.push_back(make_pair("minor", std::to_string(devProps.minor)));
  //   info.push_back(make_pair("clock_rate", std::to_string(devProps.clockRate)));
  //   info.push_back(make_pair("multi_processor_count", std::to_string(devProps.multiProcessorCount)));

  //   int runtimeVersion, driverVersion;
  //   cudaRuntimeGetVersion(&runtimeVersion);
  //   cudaDriverGetVersion(&driverVersion);
  //   info.push_back(make_pair("driver_api", std::to_string(CUDA_VERSION)));
  //   info.push_back(make_pair("driver_version", std::to_string(driverVersion)));
  //   info.push_back(
  //       make_pair("runtime_version", std::to_string(runtimeVersion)));
  //   info.push_back(
  //       make_pair("compute_version",
  //                 std::to_string((devProps.major * 10 + devProps.minor))));
  //   return info;
  // }
};

class Userinfo {
 public:
  std::vector<std::pair<string, string>> getUserinfo() const {
    std::vector<std::pair<string, string>> info;
    const char* usernotfound = "Not Found";
    if (getpwuid(getuid())) {
      info.push_back(make_pair("login", getpwuid(getuid())->pw_name));
    } else {
      info.push_back(make_pair("login", usernotfound));
    }
    return info;
  }
};

}  // namespace util
}  // namespace gunrock
