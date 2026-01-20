/**
 * @file timer.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Simple timer utility for device side code.
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/compat/runtime_api.h>

namespace gunrock {
namespace util {

struct timer_t {
  float time;

  timer_t() {
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
  }

  ~timer_t() {
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
  }

  // Reset timer by destroying and recreating events
  // This is necessary for multiple runs to prevent HIP runtime issues
  void reset() {
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
  }

  // Alias of each other, start the timer.
  // Records the start event on the specified stream (default stream 0).
  void begin(hipStream_t stream = 0) { hipEventRecord(start_, stream); }
  void start(hipStream_t stream = 0) { this->begin(stream); }

  // Alias of each other, stop the timer.
  // Records the stop event on the specified stream (default stream 0).
  float end(hipStream_t stream = 0) {
    hipEventRecord(stop_, stream);
    hipEventSynchronize(stop_);
    hipEventElapsedTime(&time, start_, stop_);

    return milliseconds();
  }
  float stop(hipStream_t stream = 0) { return this->end(stream); }

  float seconds() { return time * 1e-3; }
  float milliseconds() { return time; }

 private:
  hipEvent_t start_, stop_;
};

}  // namespace util
}  // namespace gunrock