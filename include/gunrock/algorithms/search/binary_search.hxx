/**
 * @file binary_search.cuh
 *
 * @brief
 *
 * @todo error handling within methods, remove return error_t it adds
 * unnecessary work for users to handle every return statement by each of these
 * functions.
 *
 * Maybe the file-ext needs to be .hxx instead of .cuh for faster compilation.
 *
 */

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>

namespace gunrock {

/**
 * @namespace search
 * Namespace for search algorithm, includes search on host and device.
 * Algorithms such as binary search and sorted search (see:
 * https://moderngpu.github.io/sortedsearch.html).
 */
namespace search {

enum class bound_t { upper, lower };

/**
 * @namespace binary
 * Namespace for binary search implementations on host and device.
 */
namespace binary {

// Perform binary search to find an element in an array
// Specify bounds using begin and end positions.
// Default will search the whole array of size counts.
// If duplicate items exists, this does not guarantee
// that the element found will be leftmost or rightmost element.
// XXX: Implement Search
template <typename key_pointer_t, typename key_t, typename int_t>
__host__ __device__ int_t execute(const key_pointer_t& keys,
                                  const key_t& key,
                                  int_t begin,
                                  int_t end,
                                  const bound_t bounds = bound_t::upper) {
  auto comp = [](const key_t& a, const key_t& b) { return a < b; };
  while (begin < end) {
    int_t mid = begin + ((end - begin) / 2);
    key_t key_ = keys[mid];
    bool pred = (bounds == bound_t::upper) ? !comp(key, key_) : comp(key_, key);
    if (pred)
      begin = mid + 1;
    else
      end = mid;
  }
  return begin;
}

}  // namespace binary
}  // namespace search
}  // namespace gunrock
