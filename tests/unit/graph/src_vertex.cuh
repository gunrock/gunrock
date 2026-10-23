#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/framework/operators/for/for.hxx>
#include <gunrock/io/sample.hxx>
#include <gunrock/compat/runtime_api.h>

#include <gtest/gtest.h>

using namespace gunrock;
using namespace memory;

template <typename graph_t>
void test_get_source_vertex(graph_t& G) {
  using edge_t = typename graph_t::edge_type;
  auto context =
      std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0));

  auto log_edge = [=] __device__(edge_t const& e) -> void {
    auto src = G.get_source_vertex(e);
    auto dst = G.get_destination_vertex(e);
    printf("[%d] %d -> %d\n", e, src, dst);
  };

  operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
      G, log_edge, *context);
}

TEST(graph, src_vertex) {
  auto csr = io::sample::csr<memory_space_t::device>();
  auto G = graph::build<memory_space_t::device>({}, csr);
  test_get_source_vertex(G);
}
