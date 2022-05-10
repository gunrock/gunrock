#include <gunrock/error.hxx>
#include <gunrock/graph/graph.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/cuda/cuda.hxx>
#include <gunrock/framework/operators/for/for.hxx>
#include <gunrock/io/sample_large.hxx>
#include <nvbench/nvbench.cuh>
#include <iostream>
#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/mst.hxx>

char** args;

namespace gunrock {
namespace benchmark {
void mst_bench(nvbench::state& state) {
  // Build a graph using a sample csr.
  auto csr = io::sample_large::csr();
  auto G =
      graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(csr);

  // Initialize the context.
  cuda::device_id_t device = 0;
  cuda::multi_context_t context(device);

  thrust::device_vector<weight_t> mst_weight(1);

  // --
  // GPU Run
  
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    gunrock::mst::run(G, mst_weight.data().get());
  });
}

NVBENCH_BENCH(mst_bench).set_name("mst_bench");

}  // namespace benchmark
}  // namespace gunrock
