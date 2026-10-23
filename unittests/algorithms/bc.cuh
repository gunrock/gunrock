/**
 * @file bc.cuh
 * @brief Regression tests for betweenness centrality.
 */

#include <gunrock/algorithms/bc.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/graph/graph.hxx>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

using bc_vertex_t = int;
using bc_edge_t = int;
using bc_weight_t = float;
using bc_csr_t = gunrock::format::
    csr_t<gunrock::memory_space_t::device, bc_vertex_t, bc_edge_t, bc_weight_t>;

bc_csr_t make_bc_csr(std::vector<std::vector<bc_vertex_t>> const& adjacency) {
  std::vector<bc_edge_t> row_offsets(adjacency.size() + 1, 0);
  std::vector<bc_vertex_t> column_indices;

  for (std::size_t row = 0; row < adjacency.size(); ++row) {
    column_indices.insert(column_indices.end(), adjacency[row].begin(),
                          adjacency[row].end());
    row_offsets[row + 1] = static_cast<bc_edge_t>(column_indices.size());
  }

  bc_csr_t csr(static_cast<bc_vertex_t>(adjacency.size()),
               static_cast<bc_vertex_t>(adjacency.size()),
               static_cast<bc_edge_t>(column_indices.size()));
  csr.row_offsets = row_offsets;
  csr.column_indices = column_indices;
  csr.nonzero_values =
      std::vector<bc_weight_t>(column_indices.size(), bc_weight_t{1});
  return csr;
}

std::vector<std::vector<bc_vertex_t>> make_torus(int side) {
  std::vector<std::vector<bc_vertex_t>> adjacency(side * side);
  auto vertex = [side](int row, int column) {
    row = (row + side) % side;
    column = (column + side) % side;
    return row * side + column;
  };

  for (int row = 0; row < side; ++row) {
    for (int column = 0; column < side; ++column) {
      auto& neighbors = adjacency[vertex(row, column)];
      neighbors = {vertex(row, column + 1), vertex(row, column - 1),
                   vertex(row + 1, column), vertex(row - 1, column)};
    }
  }
  return adjacency;
}

template <typename graph_t>
thrust::host_vector<bc_weight_t> run_bc(graph_t& graph,
                                        std::vector<bc_vertex_t> sources) {
  thrust::device_vector<bc_weight_t> values(graph.get_number_of_vertices(), 0);
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);

  for (auto source : sources) {
    gunrock::bc::param_t<bc_vertex_t> parameters(source);
    gunrock::bc::result_t<bc_weight_t> result(values.data().get());
    gunrock::bc::run(graph, parameters, result, context);
  }

  context->get_context(0)->synchronize();
  return values;
}

}  // namespace

TEST(algorithm, bc_torus_3x3) {
  auto csr = make_bc_csr(make_torus(3));
  gunrock::graph::graph_properties_t properties;
  properties.weighted = false;
  auto graph =
      gunrock::graph::build<gunrock::memory_space_t::device>(properties, csr);

  auto source_zero = run_bc(graph, {0});
  std::vector<bc_weight_t> source_zero_expected = {0.0f, 0.5f, 0.5f, 0.5f, 0.0f,
                                                   0.0f, 0.5f, 0.0f, 0.0f};
  for (std::size_t vertex = 0; vertex < source_zero_expected.size(); ++vertex) {
    EXPECT_FLOAT_EQ(source_zero[vertex], source_zero_expected[vertex]);
  }

  auto all_sources = run_bc(graph, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  for (auto value : all_sources) {
    EXPECT_FLOAT_EQ(value, 2.0f);
  }
}

TEST(algorithm, bc_merge_path_v2_multiple_tiles) {
  constexpr int middle_vertices = 129;
  constexpr int leaf_vertices = 4;
  constexpr int first_middle = 1;
  constexpr int first_leaf = first_middle + middle_vertices;
  constexpr int number_of_vertices = first_leaf + leaf_vertices;

  // The middle frontier has 129 rows and 516 edges, so its 645 merge items
  // cross the 640-item tile boundary.
  std::vector<std::vector<bc_vertex_t>> adjacency(number_of_vertices);
  for (int middle = 0; middle < middle_vertices; ++middle) {
    adjacency[0].push_back(first_middle + middle);
    for (int leaf = 0; leaf < leaf_vertices; ++leaf) {
      adjacency[first_middle + middle].push_back(first_leaf + leaf);
    }
  }

  auto csr = make_bc_csr(adjacency);
  gunrock::graph::graph_properties_t properties;
  properties.directed = true;
  properties.symmetric = false;
  properties.weighted = false;
  auto graph =
      gunrock::graph::build<gunrock::memory_space_t::device>(properties, csr);

  auto values = run_bc(graph, {0});
  EXPECT_FLOAT_EQ(values[0], 0.0f);
  for (int middle = 0; middle < middle_vertices; ++middle) {
    EXPECT_NEAR(values[first_middle + middle],
                2.0f / static_cast<float>(middle_vertices), 1.0e-6f);
  }
  for (int leaf = 0; leaf < leaf_vertices; ++leaf) {
    EXPECT_FLOAT_EQ(values[first_leaf + leaf], 0.0f);
  }
}
