/**
 * @file for.cu
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Unit test for for operator.
 * @date 2021-12-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <gunrock/io/sample.hxx>
#include <gunrock/framework/operators/for/for.hxx>  // for operator
#include <gunrock/graph/graph.hxx>                  // graph class

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct write_square {
  int* output;
  __device__ void operator()(int const& index) const {
    output[index] = index * index;
  }
};

struct mark_index {
  int* output;
  __device__ void operator()(int const& index) const { output[index] = 1; }
};

TEST(operators, parallel_for_visits_graph_vertices_and_frontier_elements) {
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G =
      gunrock::graph::build<gunrock::memory_space_t::device>(properties, csr);

  gunrock::gcuda::multi_context_t context(0);
  thrust::device_vector<int> graph_output(4, -1);
  gunrock::operators::parallel_for::execute<
      gunrock::operators::parallel_for_each_t::vertex>(
      G, write_square{graph_output.data().get()}, context);
  context.get_context(0)->synchronize();
  thrust::host_vector<int> host_graph_output(graph_output);
  expect_values(host_graph_output, {0, 1, 4, 9});

  gunrock::frontier::frontier_t<int, int> X;
  X.push_back(1);
  X.push_back(gunrock::numeric_limits<int>::invalid());
  X.push_back(3);
  thrust::device_vector<int> frontier_output(4, 0);
  gunrock::operators::parallel_for::execute<
      gunrock::operators::parallel_for_each_t::element>(
      X, mark_index{frontier_output.data().get()}, context);
  context.get_context(0)->synchronize();
  thrust::host_vector<int> host_frontier_output(frontier_output);
  expect_values(host_frontier_output, {0, 1, 0, 1});
}