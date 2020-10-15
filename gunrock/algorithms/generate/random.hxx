#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

namespace gunrock {
namespace algo {

namespace generate {
namespace random {
template <typename index_t, typename type_t>
void uniform_distribution(index_t begin, index_t end, type_t* input) {
  auto generate_random = [] __device__(int i) -> type_t {
    thrust::default_random_engine randEng;
    thrust::uniform_real_distribution<type_t> uniDist;
    randEng.discard(i);
    return uniDist(randEng);
  };

  thrust::transform(thrust::device, thrust::make_counting_iterator(begin),
                    thrust::make_counting_iterator(end), input + 0,
                    generate_random);
}
}  // namespace random
}  // namespace generate

}  // namespace algo
}  // namespace gunrock