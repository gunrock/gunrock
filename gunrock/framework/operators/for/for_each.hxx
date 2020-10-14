#pragma once

#include <gunrock/cuda/context.hxx>

namespace gunrock {
namespace operators {

namespace parallel_for_each {

namespace detail {
template <typename array_t, typename operator_type>
__global__ void compute(array_t array,
                        const std::size_t begin,
                        const std::size_t end,
                        operator_type op) {
  const std::size_t STRIDE = (std::size_t)blockDim.x * gridDim.x;
  for (std::size_t i = (std::size_t)blockDim.x * blockIdx.x + threadIdx.x;
       i < end; i += STRIDE)
    op(&array[i]);  // standard requires the value of array[i], but I believe
                    // reference is more useful.
}
}  // namespace detail

template <typename array_t, typename operator_type>
void execute(array_t array,
             const std::size_t begin,
             const std::size_t end,
             operator_type op,
             cuda::standard_context_t& context) {
  // XXX: context should use occupancy calculator to figure this out:
  constexpr int grid_size = 256;
  constexpr int block_size = 256;
  detail::compute<<<grid_size, block_size, 0, context.stream()>>>(array, begin,
                                                                  end, op);
}

}  // namespace parallel_for_each
}  // namespace operators
}  // namespace gunrock