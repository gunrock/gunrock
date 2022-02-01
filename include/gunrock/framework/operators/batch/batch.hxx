/**
 * @file batch.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief A batch operator is an operator that can be used to execute a
 * function/application in bactches with different inputs using C++ threads.
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

/**
 * @brief Batch operator takes a function and executes it in parallel till the
 * desired number of jobs. The function is executed from the CPU using C++
 * `std::threads`.
 *
 * @par Overview
 * The batch operator takes a function and executes it in parallel for number of
 * jobs count. This is a very rudimentary implementation of the batch operator,
 * we can expand this further by calculating the available GPU resources, and
 * also possibly using this to parallelize the batches onto multiple GPUs.
 *
 * @par Example
 * The following code snippet is a simple snippet on how to use the batch
 * operator.
 *
 * \code {.cpp}
 * // Lambda function to be executed in separate std::threads.
 * auto f = [&](std::size_t job_idx) -> float {
 *   // Function to run in batches, this can be an entire application
 *   // running on the GPU with different inputs (such as job_idx as a vertex).
 *   return run_function(G, (vertex_t)job_idx); // ... run another function.
 * };
 *
 * // Execute the batch operator on the provided lambda function.
 * operators::batch::execute(f, 10, total_elapsed.data());
 * \endcode
 *
 * @tparam function_t The function type.
 * @tparam args_t type of the arguments to the function.
 * @param f The function to execute.
 * @param number_of_jobs Number of jobs to execute.
 * @param total_elapsed pointer to an array of size 1, that stores the time
 * taken to execute all the batches.
 * @param args variadic arguments to be passed to the function (wip).
 */
template <typename function_t, typename... args_t>
void execute(function_t f,
             std::size_t number_of_jobs,
             float* total_elapsed,
             args_t&... args) {
  thrust::host_vector<float> elapsed(number_of_jobs);
  std::vector<std::thread> threads;

  for (std::size_t j = 0; j < number_of_jobs; j++) {
    threads.push_back(std::thread([&, j]() { elapsed[j] = f(j); }));
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