// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_utils.h
 *
 * @brief Utility Routines for Tests
 */

#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
    #include <time.h>
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <boost/timer/timer.hpp>
#include <gunrock/util/random_bits.h>
#include <gunrock/util/basic_utils.h>

namespace gunrock {
namespace util {

/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * CommandLineArgs interface
 */
class CommandLineArgs
{
protected:

    std::map<std::string, std::string> pairs;

public:

    // Constructor
    CommandLineArgs(int argc, char **argv)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-')) {
                continue;
            }

            std::string::size_type pos;
            std::string key, val;
            if ((pos = arg.find('=')) == std::string::npos) {
                key = std::string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = std::string(arg, 2, pos - 2);
                val = std::string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }

    // Checks whether a flag "--<flag>" is present in the commandline
    bool CheckCmdLineFlag(const char* arg_name)
    {
        std::map<std::string, std::string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end()) {
            return true;
        }
        return false;
    }

    // Returns the value specified for a given commandline parameter --<flag>=<value>
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val);

    // Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);

    // The number of pairs parsed
    int ParsedArgc()
    {
        return pairs.size();
    }
};

void DeviceInit(CommandLineArgs &args);
cudaError_t SetDevice(int dev);

template <typename T>
void CommandLineArgs::GetCmdLineArgument(
    const char *arg_name,
    T &val)
{
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
        std::istringstream str_stream(itr->second);
        str_stream >> val;
    }
}

template <typename T>
void CommandLineArgs::GetCmdLineArguments(
    const char *arg_name,
    std::vector<T> &vals)
{
     // Recover multi-value string
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {

        // Clear any default values
        vals.clear();

        std::string val_string = itr->second;
        std::istringstream str_stream(val_string);
        std::string::size_type old_pos = 0;
        std::string::size_type new_pos = 0;

        // Iterate comma-separated values
        T val;
        while ((new_pos = val_string.find(',', old_pos)) != std::string::npos) {

            if (new_pos != old_pos) {
                str_stream.width(new_pos - old_pos);
                str_stream >> val;
                vals.push_back(val);
            }

            // skip over comma
            str_stream.ignore(1);
            old_pos = new_pos + 1;
        }

        // Read last value
        str_stream >> val;
        vals.push_back(val);
    }
}

class Statistic
{
    double mean;
    double m2;
    int count;

public:
    Statistic() : mean(0.0), m2(0.0), count(0) {}

    /**
     * @brief Updates running statistic, returning bias-corrected
     * sample variance.
     *
     * Online method as per Knuth.
     *
     * @param[in] sample
     * @returns Something
     */
    double Update(double sample)
    {
        count++;
        double delta = sample - mean;
        mean = mean + (delta / count);
        m2 = m2 + (delta * (sample - mean));
        return m2 / (count - 1);                //bias-corrected
    }
};

struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return (stop - start) * 1000;
    }

/*#elif defined(CLOCK_PROCESS_CPUTIME_ID)

    timespec start;
    timespec stop;

    void Start()
    {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    }

    void Stop()
    {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    }

    float ElapsedMillis()
    {
        timespec temp;
        if ((stop.tv_nsec-start.tv_nsec)<0) {
            temp.tv_sec = stop.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+stop.tv_nsec-start.tv_nsec;
        } else {
            temp.tv_sec = stop.tv_sec-start.tv_sec;
            temp.tv_nsec = stop.tv_nsec-start.tv_nsec;
        }
        return temp.tv_nsec/1000000.0;
    }*/

#else

    /*
    rusage start;
    rusage stop;

    void Start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }
    */

    boost::timer::cpu_timer::cpu_timer cpu_t;

    void Start()
    {
        cpu_t.start();
    }

    void Stop()
    {
        cpu_t.stop();
    }

    float ElapsedMillis()
    {
        return cpu_t.elapsed().wall/1000000.0;
    }

#endif
};

// Quite simple KeyValuePair struct for doing
// Key-value sorting according to keys
template<typename A, typename B>
struct KeyValuePair
{
    A Key;
    B Value;
    bool operator<(const KeyValuePair<A, B>& rhs)
    {
        return this->Key < rhs.Key;
    }
};

} //util
} //gunrock
