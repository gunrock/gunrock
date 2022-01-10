/**
 * @file for.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Unit test for for operator.
 * @version 0.1
 * @date 2021-12-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <gunrock/io/sample.hxx>
#include <gunrock/framework/operators/for/for.hxx>  // for operator
#include <gunrock/graph/graph.hxx>                  // graph class

#include <gtest/gtest.h>

using namespace gunrock;

/// It is not valid in CUDA/NVCC to use a lambda within TEST().
/// error: The enclosing parent function ("TestBody") for an extended
/// __device__ lambda cannot have private or protected access within its class
struct f {
  __host__ __device__ void operator()(const int& v) const {}
};

TEST(operators, prallel_for) {
  // Build a sample graph using the sample csr.
  auto [csr, G] = io::sample::graph();

  // Initialize the devicecontext.
  cuda::device_id_t device = 0;
  cuda::multi_context_t context(device);

  // Launch for using a separate function.
  operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
      G, f(), context);
}