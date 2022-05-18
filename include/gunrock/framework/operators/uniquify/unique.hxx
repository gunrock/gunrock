/**
 * @file unique.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-04-15
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/framework/operators/configs.hxx>
#include <thrust/unique.h>

namespace gunrock {
namespace operators {
namespace uniquify {
namespace unique {

template <typename frontier_t>
void execute(frontier_t* input,
             frontier_t* output,
             gcuda::standard_context_t& context) {
  auto new_end = thrust::unique(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: begin
      input->end()                             // input iterator: end
  );

  auto new_size = thrust::distance(input->begin(), new_end);
  input->set_number_of_elements(new_size);

  // Simple pointer swap.
  frontier_t* temp = output;
  output = input;
  input = temp;
  temp = nullptr;
}

}  // namespace unique
}  // namespace uniquify
}  // namespace operators
}  // namespace gunrock