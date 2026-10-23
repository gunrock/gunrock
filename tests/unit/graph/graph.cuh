#include <gunrock/formats/formats.hxx>
#include <gunrock/graph/graph.hxx>

#include <gtest/gtest.h>
#include <thrust/host_vector.h>

#include <array>
#include <cmath>

namespace {

using graph_host_csr_t =
    gunrock::format::csr_t<gunrock::memory_space_t::host, int, int, float>;
using graph_device_csr_t =
    gunrock::format::csr_t<gunrock::memory_space_t::device, int, int, float>;

graph_host_csr_t make_graph_csr() {
  graph_host_csr_t csr(4, 4, 5);
  csr.row_offsets = {0, 2, 3, 5, 5};
  csr.column_indices = {1, 2, 2, 0, 3};
  csr.nonzero_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  return csr;
}

template <typename graph_t>
__global__ void inspect_graph(graph_t graph,
                              int* sources,
                              int* destinations,
                              int* degrees) {
  auto index = static_cast<int>(threadIdx.x);
  if (index < graph.get_number_of_edges()) {
    sources[index] = graph.get_source_vertex(index);
    destinations[index] = graph.get_destination_vertex(index);
  }
  if (index < graph.get_number_of_vertices()) {
    degrees[index] = graph.get_number_of_neighbors(index);
  }
}

}  // namespace

TEST(graph, properties_and_view_flags_are_composable) {
  gunrock::graph::graph_properties_t properties;
  EXPECT_FALSE(properties.directed);
  EXPECT_TRUE(properties.symmetric);
  EXPECT_TRUE(properties.weighted);

  auto views = gunrock::graph::view_t::invalid;
  views = gunrock::graph::set(views, gunrock::graph::view_t::csr);
  views = views | gunrock::graph::view_t::coo;
  EXPECT_TRUE(gunrock::graph::has(views, gunrock::graph::view_t::csr));
  EXPECT_TRUE(gunrock::graph::has(views, gunrock::graph::view_t::coo));
  EXPECT_FALSE(gunrock::graph::has(views, gunrock::graph::view_t::csc));

  views = gunrock::graph::toggle(views, gunrock::graph::view_t::csc);
  EXPECT_TRUE(gunrock::graph::has(views, gunrock::graph::view_t::csc));
  views = gunrock::graph::unset(views, gunrock::graph::view_t::coo);
  EXPECT_FALSE(gunrock::graph::has(views, gunrock::graph::view_t::coo));
}

TEST(graph, host_csr_representation_exposes_expected_edges) {
  auto csr = make_graph_csr();
  gunrock::graph::graph_properties_t properties;
  properties.directed = true;
  properties.symmetric = false;
  properties.weighted = true;
  auto graph =
      gunrock::graph::build<gunrock::memory_space_t::host>(properties, csr);

  using view_t = gunrock::graph::graph_csr_t<gunrock::memory_space_t::host, int,
                                             int, float>;
  EXPECT_EQ(graph.get_number_of_vertices(), 4);
  EXPECT_EQ(graph.get_number_of_edges(), 5);
  EXPECT_TRUE(graph.is_directed());
  EXPECT_FALSE(graph.is_symmetric());
  EXPECT_TRUE(graph.is_weighted());
  EXPECT_EQ(graph.number_of_graph_representations(), 1);
  EXPECT_TRUE(graph.template contains_representation<view_t>());

  std::array<int, 4> expected_degrees{2, 1, 2, 0};
  std::array<int, 5> expected_sources{0, 0, 1, 2, 2};
  std::array<int, 5> expected_destinations{1, 2, 2, 0, 3};
  std::array<float, 5> expected_weights{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  for (int vertex = 0; vertex < 4; ++vertex) {
    EXPECT_EQ(graph.get_number_of_neighbors(vertex), expected_degrees[vertex]);
  }
  for (int edge = 0; edge < 5; ++edge) {
    EXPECT_EQ(graph.get_source_vertex(edge), expected_sources[edge]);
    EXPECT_EQ(graph.get_destination_vertex(edge), expected_destinations[edge]);
    EXPECT_FLOAT_EQ(graph.get_edge_weight(edge), expected_weights[edge]);
    auto endpoints = graph.get_source_and_destination_vertices(edge);
    EXPECT_EQ(endpoints.source, expected_sources[edge]);
    EXPECT_EQ(endpoints.destination, expected_destinations[edge]);
  }

  EXPECT_DOUBLE_EQ(gunrock::graph::get_average_degree(graph), 1.0);
  EXPECT_NEAR(gunrock::graph::get_degree_standard_deviation(graph),
              std::sqrt(0.75), 1.0e-12);
}

TEST(graph, device_csr_accessors_match_host_reference) {
  auto host_csr = make_graph_csr();
  graph_device_csr_t device_csr(host_csr);
  auto graph =
      gunrock::graph::build<gunrock::memory_space_t::device>({}, device_csr);

  thrust::device_vector<int> sources(5, -1);
  thrust::device_vector<int> destinations(5, -1);
  thrust::device_vector<int> degrees(4, -1);
  inspect_graph<<<1, 5>>>(graph, sources.data().get(),
                          destinations.data().get(), degrees.data().get());
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  thrust::host_vector<int> host_sources(sources);
  thrust::host_vector<int> host_destinations(destinations);
  thrust::host_vector<int> host_degrees(degrees);
  expect_values(host_sources, {0, 0, 1, 2, 2});
  expect_values(host_destinations, {1, 2, 2, 0, 3});
  expect_values(host_degrees, {2, 1, 2, 0});
}
