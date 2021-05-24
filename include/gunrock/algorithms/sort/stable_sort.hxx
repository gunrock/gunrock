/**
 * @file stable_sort.cuh
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
#include <thrust/sort.h>

namespace gunrock {
namespace sort {

/**
 * @namespace stable
 * Namespace for stable sort algorithms on host and device. Supports sorting
 * keys, key-value pairs and segments. Note, std::sort in C++ uses stable sort.
 * Thrust is modeled around some of the standard libraries and therefore
 * uses/implements stable sort for the GPU based on std::sort.
 */
namespace stable {

namespace host {
// XXX: Use thrust host execution policy to perform
// sort on host?
}  // namespace host

namespace device {

// keys-only
// Stable-sort, equivalent to std::sort()
template <order_t order = order_t::ascending,
          typename key_t,
          typename storage_t,
          typename stream_t>
error_t sort(key_t* keys, int num_items) {
  error_t status = util::error::success;
  if (order == order_t::descending) {
    thrust::stable_sort(keys, keys + num_items, thrust::greater<int>());
  }

  else {
    thrust::stable_sort(keys, keys + num_items);
  }

  return status;
}

// key-value pairs
// Stable-sort key-value pairs
template <order_t order = order_t::ascending, typename key_t, typename value_t>
error_t sort_pairs(key_t* keys, value_t* values, int num_items) {
  error_t status = util::error::success;
  if (order == order_t::descending) {
    thrust::stable_sort_by_key(keys, keys + num_items, values,
                               thrust::greater<int>());
  }

  else {
    thrust::stable_sort_by_key(keys, keys + num_items, values);
  }

  return status;
}

}  // namespace device
}  // namespace stable

}  // namespace sort
}  // namespace gunrock