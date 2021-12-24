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
  // Fetch a sample csr for testing.
  auto csr = io::sample::csr();

  // Build a graph using the sample csr.
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );

  // Initialize the devicecontext.
  cuda::device_id_t device = 0;
  cuda::multi_context_t context(device);

  // Launch for using a separate function.
  operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
      G, f(), context);
}