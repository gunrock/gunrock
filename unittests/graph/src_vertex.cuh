#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/framework/operators/for/for.hxx>
#include <gunrock/io/sample.hxx>

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

int main(int argc, char** argv) {
  auto [csr, G] = io::sample::graph();
  test_get_source_vertex(G);
}
