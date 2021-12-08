#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

namespace gunrock {
namespace generate {
namespace random {

/**
 * @brief generate random numbers between begin and end for a given vector.
 *
 * @param input vector (thrust::device or host vectors work fine.)
 * @param begin (default = 0.0f) random number range starting value.
 * @param end (default = 1.0f) random number range ending value.
 */
template <typename vector_t, typename type_t = typename vector_t::value_type>
void uniform_distribution(vector_t& input,
                          type_t begin = 0.0f,
                          type_t end = 1.0f) {
  auto generate_random = [=] __host__ __device__(int i) -> type_t {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<type_t> uniform(begin, end);
    rng.discard(i);
    return uniform(rng);
  };

  thrust::transform(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(input.size()),
                    input.begin(), generate_random);
}

}  // namespace random
}  // namespace generate
}  // namespace gunrock