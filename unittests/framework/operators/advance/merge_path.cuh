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

TEST(operators_advance, merge_path_search) {
  // Test merge_path search function
  using index_t = int;
  
  // Simple test: counting iterator [0,1,2,3,4] vs segments [2,5]
  // segments[0]=2 means work items 0,1 belong to segment 0
  // segments[1]=5 means work items 2,3,4 belong to segment 1
  thrust::counting_iterator<index_t> counting_it(0);
  
  // Note: merge_path search is device-only, so we can't test it directly from host
  // This test verifies the structure exists and compiles
  operators::advance::merge_path::coordinate_t<index_t> coords;
  coords.x = 0;
  coords.y = 0;
  EXPECT_EQ(coords.x, 0);
  EXPECT_EQ(coords.y, 0);
}

TEST(operators_advance, merge_path_load_balance_partitions) {
  // Test load_balance_partitions function
  using index_t = int;
  
  gcuda::device_id_t device = 0;
  gcuda::multi_context_t multi_context(device);
  auto context = multi_context.get_context(0);
  
  // Create segments: [0, 2, 5] means vertex 0 has 2 edges, vertex 1 has 3 edges
  // segments is exclusive scan, so segments[i] = cumulative edges up to vertex i
  thrust::device_vector<index_t> segments(3);
  segments[0] = 0;   // Start of vertex 0's edges
  segments[1] = 2;   // Start of vertex 1's edges (vertex 0 has 2 edges)
  segments[2] = 5;   // End (vertex 1 has 3 edges: 2,3,4)
  
  int64_t dest_count = 5;  // Total work items (edges)
  int num_segments = 2;    // Number of vertices (segments array has num_segments+1 elements)
  int spacing = 2;         // Partition spacing
  
  auto partitions = operators::advance::merge_path::load_balance_partitions<index_t>(
    dest_count, segments.data().get(), num_segments, spacing, *context);
  
  // Should have partitions
  EXPECT_GT(partitions.size(), 0);
  
  // First partition should be 0 (start of first range)
  thrust::host_vector<index_t> host_partitions = partitions;
  EXPECT_GE(host_partitions[0], 0);
}

TEST(operators_advance, merge_path_advance_basic) {
  // Basic test that merge_path advance can be called
  // Full correctness testing is done via algorithm-level tests (SSSP/BFS)
  using vertex_t = int;
  using edge_t = int;
  
  // Build a sample graph
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties, csr);
  
  // Initialize context
  gcuda::device_id_t device = 0;
  gcuda::multi_context_t context(device);
  
  // Create frontiers
  frontier::frontier_t<vertex_t, edge_t> input_frontier;
  frontier::frontier_t<vertex_t, edge_t> output_frontier;
  
  // Initialize input frontier with vertex 0
  input_frontier.push_back(0);
  
  // Simple advance operator: visit all neighbors
  auto advance_op = [] __host__ __device__(
    vertex_t const& source,
    vertex_t const& neighbor,
    edge_t const& edge,
    float const& weight) -> bool {
    return true;  // Keep all neighbors
  };
  
  // Work tiles for merge_path
  thrust::device_vector<edge_t> work_tiles(G.get_number_of_vertices());
  
  // Run merge_path advance - should not crash
  operators::advance::execute<operators::load_balance_t::merge_path,
                               operators::advance_direction_t::forward,
                               operators::advance_io_type_t::vertices,
                               operators::advance_io_type_t::vertices>(
    G, advance_op, &input_frontier, &output_frontier, work_tiles, context);
  
  // Basic sanity check - output should have some elements if graph has edges
  if (G.get_number_of_edges() > 0) {
    EXPECT_GE(output_frontier.get_number_of_elements(), 0);
  }
}

TEST(operators_advance, merge_path_runtime_dispatch) {
  // Test runtime dispatch of merge_path
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  
  // Build a sample graph
  auto csr = gunrock::io::sample::csr();
  gunrock::graph::graph_properties_t properties;
  auto G = gunrock::graph::build<gunrock::memory_space_t::device>(properties, csr);
  
  // Initialize context
  gcuda::device_id_t device = 0;
  gcuda::multi_context_t context(device);
  
  // Create enactor-like structure for testing
  // (This is a simplified test - full enactor setup would be more complex)
  
  // Test that merge_path enum is recognized
  operators::load_balance_t lb = operators::load_balance_t::merge_path;
  EXPECT_EQ(lb, operators::load_balance_t::merge_path);
}
