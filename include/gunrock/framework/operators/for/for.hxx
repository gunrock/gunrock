#pragma once

#include <gunrock/cuda/context.hxx>

namespace gunrock {
namespace operators {

namespace parallel_for {

namespace detail {
template <typename operator_type>
__global__ void compute(const std::size_t begin,
                        const std::size_t end,
                        operator_type op) {
  const std::size_t STRIDE = (std::size_t)blockDim.x * gridDim.x;
  for (std::size_t i = (std::size_t)blockDim.x * blockIdx.x + threadIdx.x;
       i < end; i += STRIDE)
    op(i);
}
}  // namespace detail

template <typename operator_type>
void execute(const std::size_t begin,
             const std::size_t end,
             operator_type op,
             cuda::multi_context_t& context) {
  // XXX: context should use occupancy calculator to figure this out:
  constexpr int grid_size = 256;
  constexpr int block_size = 256;

  if (context.size() == 1) {
    auto context0 = context.get_context(0);
    detail::compute<<<grid_size, block_size, 0, context0->stream()>>>(begin,
                                                                      end, op);
  } else {
    error::throw_if_exception(cudaErrorUnknown,
                              "`context.size() != 1` not supported");
  }
}

}  // namespace parallel_for
}  // namespace operators
}  // namespace gunrock