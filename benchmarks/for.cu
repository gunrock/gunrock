#include <gunrock/error.hxx>
#include <gunrock/graph/graph.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/cuda/cuda.hxx>
#include <gunrock/framework/operators/for/for.hxx>
#include <gunrock/util/sample.hxx>

#include <nvbench/nvbench.cuh>

namespace gunrock {
namespace benchmark {
void parallel_for(nvbench::state& state) {
  auto csr = sample::csr();

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );

  // Initialize the context.
  cuda::device_id_t device = 0;
  cuda::multi_context_t context(device);

  auto f = [=] __device__(int const& v) -> void { printf("%i\n", v); };

  state.exec([&](nvbench::launch& launch) {
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
