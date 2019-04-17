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
#undef small  // Windows is terrible for polluting macro namespace
#elif defined(CLOCK_PROCESS_CPUTIME_ID)
#include <sys/time.h>
#else
#include <sys/resource.h>
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <cassert>
#include <map>
#include <string>
#include <vector>
#include <stack>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
//#include <boost/timer/timer.hpp>
#include <chrono>
#include <gunrock/util/random_bits.h>
#include <gunrock/util/basic_utils.h>

// #include <gunrock/util/gitsha1.cpp>

namespace gunrock {
namespace util {

/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * CommandLineArgs interface
 */
class CommandLineArgs {
 private:
  int argc;
  char **argv;

 protected:
  std::map<std::string, std::string> pairs;

 public:
  // Constructor
  CommandLineArgs(int _argc, char **_argv) : argc(_argc), argv(_argv) {
    for (int i = 1; i < argc; i++) {
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
  bool CheckCmdLineFlag(const char *arg_name) {
    std::map<std::string, std::string>::iterator itr;
    if ((itr = pairs.find(arg_name)) != pairs.end()) {
      return true;
    }
    return false;
  }

  // Returns the value specified for a given commandline
  // parameter --<flag>=<value>
  template <typename T>
  void GetCmdLineArgument(const char *arg_name, T &val);

  // Returns the values specified for a given commandline
  // parameter --<flag>=<value>,<value>*
  template <typename T>
  void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);

  // The number of pairs parsed
  int ParsedArgc() { return pairs.size(); }

  std::string GetEntireCommandLine() const {
    std::string commandLineStr = "";
    for (int i = 0; i < argc; i++) {
      commandLineStr.append(
          std::string(argv[i]).append((i < argc - 1) ? " " : ""));
    }
    return commandLineStr;
  }

  template <typename T>
  void ParseArgument(const char *name, T &val) {
    if (CheckCmdLineFlag(name)) {
      GetCmdLineArgument(name, val);
    }
  }

  char *GetCmdLineArgvGraphType() {
    char *graph_type = argv[1];
    return graph_type;
  }

  char *GetCmdLineArgvDataset() {
    char *market_filename;
    size_t graph_args = argc - pairs.size() - 1;
    market_filename = (graph_args == 2) ? argv[2] : NULL;
    return market_filename;
  }

  char *GetCmdLineArgvQueryDataset() {
    char *market_fileName;
    size_t graph_args = argc - pairs.size() - 1;
    market_fileName = (graph_args > 1) ? argv[2] : NULL;
    return market_fileName;
  }

  char *GetCmdLineArgvDataDataset() {
    char *market_fileName;
    size_t graph_args = argc - pairs.size() - 1;
    market_fileName =
        (graph_args > 2)
            ? ((graph_args == 5) ? argv[graph_args - 1] : argv[graph_args])
            : NULL;
    return market_fileName;
  }

  char *GetCmdLineArgvQueryLabel() {
    char *label_fileName;
    size_t graph_args = argc - pairs.size() - 1;
    label_fileName = (graph_args == 5) ? argv[3] : NULL;
    return label_fileName;
  }

  char *GetCmdLineArgvDataLabel() {
    char *label_fileName;
    size_t graph_args = argc - pairs.size() - 1;
    label_fileName = (graph_args == 5) ? argv[graph_args] : NULL;
    return label_fileName;
  }
};

void DeviceInit(CommandLineArgs &args);
cudaError_t SetDevice(int dev);

template <typename T>
void CommandLineArgs::GetCmdLineArgument(const char *arg_name, T &val) {
  std::map<std::string, std::string>::iterator itr;
  if ((itr = pairs.find(arg_name)) != pairs.end()) {
    std::istringstream str_stream(itr->second);
    str_stream >> val;
  }
}

template <typename T>
void CommandLineArgs::GetCmdLineArguments(const char *arg_name,
                                          std::vector<T> &vals) {
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

class Statistic {
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
  double Update(double sample) {
    count++;
    double delta = sample - mean;
    mean = mean + (delta / count);
    m2 = m2 + (delta * (sample - mean));
    return m2 / (count - 1);  // bias-corrected
  }
};

struct CpuTimer {
#if defined(_WIN32) || defined(_WIN64)

  LARGE_INTEGER ll_freq;
  LARGE_INTEGER ll_start;
  LARGE_INTEGER ll_stop;

  CpuTimer() { QueryPerformanceFrequency(&ll_freq); }

  void Start() { QueryPerformanceCounter(&ll_start); }

  void Stop() { QueryPerformanceCounter(&ll_stop); }

  double TimeDifference(LARGE_INTEGER &a, LARGE_INTEGER &b) {
    double start = double(a.QuadPart) / double(ll_freq.QuadPart);
    double stop = double(b.QuadPart) / double(ll_freq.QuadPart);

    return (stop - start) * 1000;
  }

  double ElapsedMillis() { return TimeDifference(ll_start, ll_stop); }

  double MillisSinceStart() {
    LARGE_INTEGER ll_current;
    QueryPerformanceCounter(&ll_current);
    return TimeDifference(ll_start, ll_current);
  }

#elif defined(CLOCK_PROCESS_CPUTIME_ID)

  double start;
  double stop;

  double GetCurrentTime() {
    static struct timeval tv;
    static struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec + 1.e-6 * tv.tv_usec;
  }

  void Start() { start = GetCurrentTime(); }

  void Stop() { stop = GetCurrentTime(); }

  double TimeDifference(double &a, double &b) { return 1000 * (b - a); }

  double ElapsedMillis() { return TimeDifference(start, stop); }

  double MillisSinceStart() {
    double current = GetCurrentTime();
    return TimeDifference(start, current);
  }

/*#elif defined(boost)

  boost::timer::cpu_timer::cpu_timer cpu_t;

  void Start() { cpu_t.start(); }

  void Stop() { cpu_t.stop(); }

  double ElapsedMillis() { return cpu_t.elapsed().wall / 1000000.0; }

    double MillisSinceStart()
    {
        return cpu_t.elapsed().wall / 1000000.0;
    }
*/
#else // C++11 timer
    typedef std::chrono::time_point<std::chrono::system_clock> TimeT;
    TimeT start, end;

    void Start()
    {
        start = std::chrono::system_clock::now();
    }

    void Stop()
    {
        end = std::chrono::system_clock::now();
    }

    double TimeDifference(TimeT &start, TimeT &end)
    {
        std::chrono::duration<double> diff = end - start;
        return std::chrono::milliseconds(diff).count();
    }

    double MillisSinceStart()
    {
        auto time_now = std::chrono::system_clock::now();
        return TimeDifference(start, time_now);
    }

    double ElapsedMillis()
    {
        return TimeDifference(start, end);
    }
#endif
};

// Quite simple KeyValuePair struct for doing
// Key-value sorting according to keys
template <typename A, typename B>
struct KeyValuePair {
  A Key;
  B Value;
  bool operator<(const KeyValuePair<A, B> &rhs) { return this->Key < rhs.Key; }
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
