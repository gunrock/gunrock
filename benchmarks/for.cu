#include <gunrock/error.hxx>
#include <gunrock/graph/graph.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/cuda/cuda.hxx>
#include <gunrock/framework/operators/for/for.hxx>
#include <gunrock/io/sample.hxx>

#include <nvbench/nvbench.cuh>

namespace gunrock {
namespace benchmark {
void parallel_for(nvbench::state& state) {
  // Build a graph using a sample csr.
  auto csr = io::sample::csr();
  auto G =
      graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(csr);

  // Initialize the context.
  gcuda::device_id_t device = 0;
  gcuda::multi_context_t context(device);

  vector_t<int> vertices(G.get_number_of_vertices());
  auto d_vertices = vertices.data().get();

  auto f = [=] __device__(int const& v) -> void { d_vertices[v] = v; };

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,       // graph
        f,       // lambda function
        context  // context
    );
  });
}

NVBENCH_BENCH(parallel_for).set_name("parallel_for");

}  // namespace benchmark
}  // namespace gunrock
