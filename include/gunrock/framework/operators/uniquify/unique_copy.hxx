/**
 * @file unique_copy.hxx
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
namespace unique_copy {

template <typename frontier_t>
void execute(frontier_t* input,
             frontier_t* output,
             gcuda::standard_context_t& context) {
  // Make sure output frontier has enough space.
  if (output->get_capacity() < input->get_number_of_elements())
    output->reserve(input->get_number_of_elements());

  auto new_end = thrust::unique_copy(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: begin
      input->end(),                            // input iterator: end
      output->begin()                          // output iterator: begin
  );

  auto new_size = thrust::distance(input->begin(), new_end);
  output->set_number_of_elements(new_size);
}

}  // namespace unique_copy
}  // namespace uniquify
}  // namespace operators
}  // namespace gunrock