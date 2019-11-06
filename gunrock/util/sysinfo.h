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

#include <gunrock/util/json_spirit_writer_template.h>
#include <sys/utsname.h>        /* for Cpuinfo */
#include <cuda.h>               /* for Gpuinfo */
#include <cuda_runtime_api.h>   /* for Gpuinfo */
#include <pwd.h>                /* for Userinfo */


namespace gunrock {
namespace util {

class Sysinfo {
private:
    struct utsname uts;
public:
    Sysinfo()
    {
        uname(&uts);
    }

    std::string sysname() const
    {
        return std::string(uts.sysname);
    }
    std::string release() const
    {
        return std::string(uts.release);
    }
    std::string version() const
    {
        return std::string(uts.version);
    }
    std::string machine() const
    {
        return std::string(uts.machine);
    }
    std::string nodename() const
    {
        return std::string(uts.nodename);
    }

    json_spirit::mObject getSysinfo() const
    {
        json_spirit::mObject json_sysinfo;

        json_sysinfo["sysname"] = sysname();
        json_sysinfo["release"] = release();
        json_sysinfo["version"] = version();
        json_sysinfo["machine"] = machine();
        json_sysinfo["nodename"] = nodename();
        return json_sysinfo;
    }

};

class Gpuinfo {
public:
    json_spirit::mObject getGpuinfo() const
    {
        json_spirit::mObject info;
        cudaDeviceProp devProps;

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)   /* no valid devices */
        {
            return info;        /* empty */
        }
        int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&devProps, dev);
        info["name"] = devProps.name;
        info["total_global_mem"] = int64_t(devProps.totalGlobalMem);
        info["major"] = devProps.major;
        info["minor"] = devProps.minor;
        info["clock_rate"] = devProps.clockRate;
        info["multi_processor_count"] = devProps.multiProcessorCount;

        int runtimeVersion, driverVersion;
        cudaRuntimeGetVersion(&runtimeVersion);
        cudaDriverGetVersion(&driverVersion);
        info["driver_api"] = CUDA_VERSION;
        info["driver_version"] = driverVersion;
        info["runtime_version"] = runtimeVersion;
        info["compute_version"] = devProps.major * 10 + devProps.minor;
        return info;
    }
};

class Userinfo {
public:
    json_spirit::mObject getUserinfo() const
    {
        json_spirit::mObject info;
        const char * usernotfound = "Not Found";
        if (getpwuid(getuid()))
        {
            info["login"] = getpwuid(getuid())->pw_name;
        } else
        {
            info["login"] = usernotfound;
        }
        return info;
    }
};

} //util
} //gunrock
