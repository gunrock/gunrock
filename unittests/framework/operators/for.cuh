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

/// It is not valid in CUDA/NVCC to use a lambda within TEST().
/// error: The enclosing parent function ("TestBody") for an extended
/// __device__ lambda cannot have private or protected access within its class
struct f {
  __host__ __device__ void operator()(const int& v) const {}
};

TEST(operators, prallel_for) {
  // Build a sample graph using the sample csr.
  auto csr = gunrock::io::sample::csr();
  auto G = gunrock::graph::build::from_csr<gunrock::memory_space_t::device,
                                           gunrock::graph::view_t::csr>(csr);

  // Initialize the devicecontext.
  gunrock::gcuda::device_id_t device = 0;
  gunrock::gcuda::multi_context_t context(device);

  // Launch for using a separate function.
  gunrock::operators::parallel_for::execute<
      gunrock::operators::parallel_for_each_t::vertex>(G, f(), context);

  // Build a sample frontier.
  gunrock::frontier::frontier_t<int, int> X;
  X.push_back(1);

  // Launch for on a frontier.
  gunrock::operators::parallel_for::execute<
      gunrock::operators::parallel_for_each_t::element>(X, f(), context);
}