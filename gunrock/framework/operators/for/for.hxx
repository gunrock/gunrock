#pragma once

#include <gunrock/cuda/context.hxx>

namespace gunrock {
namespace operators {

namespace parallel_for {

template <typename operator_type>
__global__ void compute(const std::size_t begin,
                        const std::size_t end,
                        operator_type op) {
  const std::size_t STRIDE = (std::size_t)blockDim.x * gridDim.x;
  for (std::size_t i = (std::size_t)blockDim.x * blockIdx.x + threadIdx.x;
       i < s; i += STRIDE)
    op(i);
}

template <typename operator_type>
void execute(const std::size_t begin,
             const std::size_t end,
             operator_type op,
             standard_context_t& context) {
  // XXX: context should use occupancy calculator to figure this out:
  constexpr int grid_size = 256;
  constexpr int block_size = 256;
  compute<<<grid_size, block_size, 0, context.stream()>>>(begin, end, op);
}

}  // namespace parallel_for
}  // namespace operators
}  // namespace gunrock