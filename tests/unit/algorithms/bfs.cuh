#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/graph/graph.hxx>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <limits>
#include <memory>
#include <vector>

TEST(algorithm, bfs_matches_reference_distances) {
  using device_csr_t =
      gunrock::format::csr_t<gunrock::memory_space_t::device, int, int, float>;

  device_csr_t csr(6, 6, 7);
  csr.row_offsets = std::vector<int>{0, 2, 3, 5, 6, 7, 7};
  csr.column_indices = std::vector<int>{1, 2, 3, 3, 4, 5, 5};
  csr.nonzero_values = std::vector<float>(7, 1.0f);

  gunrock::graph::graph_properties_t properties;
  properties.directed = true;
  properties.symmetric = false;
  properties.weighted = false;
  auto graph =
      gunrock::graph::build<gunrock::memory_space_t::device>(properties, csr);
  auto context = std::make_shared<gunrock::gcuda::multi_context_t>(0);

  auto const unreachable = std::numeric_limits<int>::max();
  struct case_t {
    int source;
    std::vector<int> expected;
  };
  std::vector<case_t> cases{
      {0, {0, 1, 1, 2, 2, 3}},
      {4, {unreachable, unreachable, unreachable, unreachable, 0, 1}},
  };

  for (auto const& test_case : cases) {
    thrust::device_vector<int> distances(6);
    gunrock::options_t options;
    options.enable_filter = true;
    gunrock::bfs::param_t<int> parameters(test_case.source, options);
    gunrock::bfs::result_t<int> result(distances.data().get(), nullptr);

    gunrock::bfs::run(graph, parameters, result, context);
    context->get_context(0)->synchronize();
    thrust::host_vector<int> actual(distances);

    ASSERT_EQ(actual.size(), test_case.expected.size());
    for (std::size_t vertex = 0; vertex < test_case.expected.size(); ++vertex) {
      EXPECT_EQ(actual[vertex], test_case.expected[vertex])
          << "source " << test_case.source << ", vertex " << vertex;
    }
  }
}
