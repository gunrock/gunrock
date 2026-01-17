/**
 * @file merge_path.cuh
 * @brief Unit test for merge_path load balancing advance operator.
 * @date 2026-01-16
 */

#include <gunrock/framework/operators/advance/merge_path.hxx>
#include <gunrock/framework/operators/advance/advance.hxx>
#include <gunrock/framework/operators/advance/block_mapped.hxx>
#include <gunrock/framework/frontier/vector_frontier.hxx>
#include <gunrock/graph/graph.hxx>
#include <gunrock/io/sample.hxx>
#include <gunrock/cuda/context.hxx>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace gunrock;
using namespace memory;

TEST(operators_advance, merge_path_coordinate_struct) {
  // Test that the coordinate_t struct exists and works
  using index_t = int;
  
  operators::advance::merge_path::coordinate_t<index_t> coords;
  coords.x = 5;
  coords.y = 10;
  EXPECT_EQ(coords.x, 5);
  EXPECT_EQ(coords.y, 10);
}

TEST(operators_advance, merge_path_enum_exists) {
  // Test that merge_path enum is recognized in load_balance_t
  operators::load_balance_t lb = operators::load_balance_t::merge_path;
  EXPECT_EQ(lb, operators::load_balance_t::merge_path);
  
  // Verify it's different from other load balance types
  EXPECT_NE(lb, operators::load_balance_t::block_mapped);
  EXPECT_NE(lb, operators::load_balance_t::thread_mapped);
}
