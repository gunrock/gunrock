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

namespace gunrock {
namespace util {

struct timer_t {
  float time;

  timer_t() {
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
    hipEventRecord(start_);
  }

  ~timer_t() {
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
  }

  // Alias of each other, start the timer.
  void begin() { hipEventRecord(start_); }
  void start() { this->begin(); }

  // Alias of each other, stop the timer.
  float end() {
    hipEventRecord(stop_);
    hipEventSynchronize(stop_);
    hipEventElapsedTime(&time, start_, stop_);

    return milliseconds();
  }
  float stop() { return this->end(); }

  float seconds() { return time * 1e-3; }
  float milliseconds() { return time; }

 private:
  hipEvent_t start_, stop_;
};

}  // namespace util
}  // namespace gunrock