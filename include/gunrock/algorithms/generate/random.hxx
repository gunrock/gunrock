#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

namespace gunrock {
namespace algo {

namespace generate {
namespace random {

template <typename index_t, typename iterator_t>
void uniform_distribution(index_t begin, index_t end, iterator_t input) {
  using type_t = typename std::iterator_traits<iterator_t>::value_type;

  auto generate_random = [] __device__(int i) -> type_t {
    thrust::default_random_engine rng;
    /* thrust::uniform_real_distribution<type_t> uni_dist; */
    rng.discard(i);
    return rng(); /* uni_dist(rng); */
  };

  thrust::transform(thrust::make_counting_iterator(begin),
                    thrust::make_counting_iterator(end), input,
                    generate_random);
}
}  // namespace random
}  // namespace generate

}  // namespace algo
}  // namespace gunrock