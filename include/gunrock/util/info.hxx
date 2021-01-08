/**
 * @file
 * log.hxx
 *
 * @brief Running statistic collector with json
 */

#pragma once

#include <cmath>
#include <cstdio>
#include <ctime>
#include <time.h>
#include <vector>

// RapidJSON includes (required)
#include <rapidjson/document.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include <gunrock/util/error.hxx>
#include <gunrock/util/gitsha1.hxx>

namespace gunrock {
namespace util {

/**
 * @namespace stats
 *
 */
namespace stats {

/**
 * @brief log data structure to build gunrock's application related statistics.
 * Outputs to stdout or a json file.
 */
class log {
  // Timing
  double _preprocess_time;
  double _postprocess_time;

  double _total_time;

  double _total_algo_time;                  // sum of running times
  double _avg_algo_time;                    // average of running times
  double _max_algo_time;                    // maximum running time
  double _min_algo_time;                    // minimum running time
  double _stddev_algo_time;                 // std. deviation of running times
  std::vector<double> algo_times;           // array of running times (raw)
  std::vector<double> filtered_algo_times;  // array of running times (filtered)
  int num_runs;                             // number of runs

  // Graph Statistics
  int64_t num_nodes;
  int64_t num_edges;
  int average_degree;

  double max_m_teps;  // maximum MTEPS
  double min_m_teps;  // minimum MTEPS
  double m_teps;

  // Traversal Statistics
  int64_t nodes_queued;
  int64_t edges_queued;
  int64_t nodes_visited;
  int64_t edges_visited;

  int64_t search_depth;
  double avg_duty;

  // Redundant data computed
  double nodes_redundance;
  double edges_redundance;

 public:
  /**
   * @brief Construct a new log object
   *
   */
  log() {}

  /**
   * @brief logging individual runs
   *
   * @param single_elapsed a single elapsed time of the algorithm
   */
  void collect_single_run(double single_elapsed) {
    _total_algo_time += single_elapsed;
    algo_times.push_back(single_elapsed);
    ++num_runs;
  }

  void traversal_stats() {}

  void graph_stats(graph_t graph) {}

  void compute_common_stats() {}

};  // class log

}  // namespace stats
}  // namespace util
}  // namespace gunrock