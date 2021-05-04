/**
 * @file batch.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-05-04
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/cuda/context.hxx>

#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <thread>

namespace gunrock {
namespace operators {
namespace batch {

template <typename function_t, typename... args_t>
void execute(function_t f,
             std::size_t batch_size,
             float* total_elapsed,
             args_t&... args) {
  thrust::host_vector<float> elapsed(batch_size);
  std::vector<std::thread> threads;

  for (std::size_t idx = 0; idx < batch_size; idx++) {
    threads.push_back(std::thread([&, idx]() { elapsed[idx] = f(idx); }));
  }

  for (auto& thread : threads)
    thread.join();

  total_elapsed[0] =
      thrust::reduce(thrust::host, elapsed.begin(), elapsed.end(), (float)0.0f,
                     thrust::plus<float>());
}

}  // namespace batch
}  // namespace operators
}  // namespace gunrock