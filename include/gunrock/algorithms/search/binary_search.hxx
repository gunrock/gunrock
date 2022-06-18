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
    int_t mid = (begin + end) / 2;
    key_t key_ = keys[mid];
    bool pred = (bounds == bound_t::upper) ? !comp(key, key_) : comp(key_, key);
    if (pred)
      begin = mid + 1;
    else
      end = mid;
  }
  return begin;
}

// Find the leftmost element in an array.
// Specify bounds using begin and end positions.
// Default will search the whole array of size counts.
// Guarantees that the element found will be the one at the
// corner (left or right most position) of the array.
template <typename iterator_t, typename key_t, typename comp_t>
__host__ __device__ iterator_t lower_bound(
    iterator_t first,
    iterator_t last,
    const key_t& value,
    comp_t comp = [] __host__ __device__(const key_t& pivot, const key_t& key) {
      return pivot < key;  // less_than_comparable
    }) {
  return thrust::lower_bound(thrust::seq, first, last, value, comp);
}

template <typename key_pointer_t, typename key_t, typename index_t>
__host__ __device__ key_t lower_bound(const key_pointer_t keys,
                                      const key_t& key,
                                      const index_t size) {
  auto it = search::binary::lower_bound(
      thrust::counting_iterator<key_t>(0),
      thrust::counting_iterator<key_t>(size), key,
      [keys] __host__ __device__(const key_t& pivot, const key_t& key) {
        return keys[pivot] < key;
      });
  return *it;
}

// Find the rightmost element in an array.
// Specify bounds using begin and end positions.
// Default will search the whole array of size counts.
// Guarantees that the element found will be the one at the
// corner (left or right most position) of the array.
template <typename iterator_t, typename key_t, typename comp_t>
__host__ __device__ iterator_t upper_bound(
    iterator_t first,
    iterator_t last,
    const key_t& value,
    comp_t comp = [] __host__ __device__(const key_t& pivot, const key_t& key) {
      return !(key < pivot);  // greater_than_comparable
    }) {
  return thrust::upper_bound(thrust::seq, first, last, value, comp);
}

template <typename key_pointer_t, typename key_t, typename index_t>
__host__ __device__ key_t upper_bound(const key_pointer_t keys,
                                      const key_t& key,
                                      const index_t size) {
  auto it = search::binary::upper_bound(
      thrust::counting_iterator<key_t>(0),
      thrust::counting_iterator<key_t>(size), key,
      [keys] __host__ __device__(const key_t& pivot, const key_t& key) {
        return keys[pivot] < key;
      });
  return *it;
}

template <typename key_t, typename index_t>
__host__ __device__ index_t rightmost(const key_t* keys,
                                      const key_t& key,
                                      const index_t count) {
  index_t begin = 0;
  index_t end = count;
  while (begin < end) {
    index_t mid = floor((begin + end) / 2);
    key_t key_ = keys[mid];
    bool pred = key_ > key;
    if (pred) {
      end = mid;
    } else {
      begin = mid + 1;
    }
  }
  return end - 1;
}

}  // namespace binary
}  // namespace search
}  // namespace gunrock
