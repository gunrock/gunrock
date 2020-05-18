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

namespace gunrock {
namespace algo {

/**
 * @namespace search
 * Namespace for search algorithm, includes search on host and device.
 * Algorithms such as binary search and sorted search (see:
 * https://moderngpu.github.io/sortedsearch.html).
 */
namespace search {

enum bound_t
{
  upper,
  lower
};

/**
 * @namespace binary
 * Namespace for binary search implementations on host and device.
 */
namespace binary
{
  namespace host {

  // Perform binary search to find an element in an array
  // Specify bounds using begin and end positions.
  // Default will search the whole array of size counts.
  // If duplicate items exists, this does not guarantee
  // that the element found will be leftmost or rightmost element.
  template<bound_t bounds = bound_t::upper,
           typename key_t,
           typename int_t,
           typename comp_t>
  __host__ int_t search(const key_t* keys,
                        const key_t key,
                        int_t count,
                        int_t begin = 0,
                        int_t end = count,
                        comp_t comp = [](const key_t& a, const key_t& b) {
                          return a < b;
                        })
  {
    while (begin < end) {
      int_t mid = (begin + end) / 2;
      key_t key_ = keys[mid];
      bool pred =
        (bounds == bound_t::upper) ? !comp(key, key_) : comp(key_, key);
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
  template<bound_t bounds = bound_t::upper,
           typename key_t,
           typename int_t,
           typename comp_t>
  __host__ int_t leftmost(const key_t* keys,
                          const key_t key,
                          int_t count,
                          int_t begin = 0,
                          int_t end = count,
                          comp_t less = [](const key_t& a, const key_t& b) {
                            return a < b;
                          })
  {
    while (begin < end) {
      int_t mid = floor((begin + end) / 2);
      key_t key_ = keys[mid];
      bool pred = less(key_, key);
      if (pred) {
        begin = mid + 1;
      } else {
        end = mid;
      }
    }
    return begin;
  }

  // Find the rightmost element in an array.
  // Specify bounds using begin and end positions.
  // Default will search the whole array of size counts.
  // Guarantees that the element found will be the one at the
  // corner (left or right most position) of the array.
  template<bound_t bounds = bound_t::upper,
           typename key_t,
           typename int_t,
           typename comp_t>
  __host__ int_t rightmost(const key_t* keys,
                           const key_t key,
                           int_t count,
                           int_t begin = 0,
                           int_t end = count,
                           comp_t greater = [](const key_t& a, const key_t& b) {
                             return a > b;
                           })
  {
    while (begin < end) {
      int_t mid = floor((begin + end) / 2);
      key_t key_ = keys[mid];
      bool pred = greater(key_, key);
      if (pred) {
        end = mid;
      } else {
        begin = mid + 1;
      }
    }
    return end - 1;
  }

  } // namespace: host

  namespace device {
  namespace block {

  // Perform binary search to find an element in an array
  // Specify bounds using begin and end positions.
  // Default will search the whole array of size counts.
  // If duplicate items exists, this does not guarantee
  // that the element found will be leftmost or rightmost element.
  template<bound_t bounds = bound_t::upper,
           typename key_t,
           typename int_t,
           typename comp_t>
  __device__ int_t search(const key_t* keys,
                          const key_t key,
                          int_t count,
                          int_t begin = 0,
                          int_t end = count,
                          comp_t comp = [](const key_t& a, const key_t& b) {
                            return a < b;
                          })
  {
    while (begin < end) {
      int_t mid = (begin + end) / 2;
      key_t key_ = keys[mid];
      bool pred =
        (bounds == bound_t::upper) ? !comp(key, key_) : comp(key_, key);
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
  template<bound_t bounds = bound_t::upper,
           typename key_t,
           typename int_t,
           typename comp_t>
  __device__ int_t leftmost(const key_t* keys,
                            const key_t key,
                            int_t count,
                            int_t begin = 0,
                            int_t end = count,
                            comp_t less = [](const key_t& a, const key_t& b) {
                              return a < b;
                            })
  {
    while (begin < end) {
      int_t mid = floor((begin + end) / 2);
      key_t key_ = keys[mid];
      bool pred = less(key_, key);
      if (pred) {
        begin = mid + 1;
      } else {
        end = mid;
      }
    }
    return begin;
  }

  // Find the rightmost element in an array.
  // Specify bounds using begin and end positions.
  // Default will search the whole array of size counts.
  // Guarantees that the element found will be the one at the
  // corner (left or right most position) of the array.
  template<bound_t bounds = bound_t::upper,
           typename key_t,
           typename int_t,
           typename comp_t>
  __device__ int_t rightmost(const key_t* keys,
                             const key_t key,
                             int_t count,
                             int_t begin = 0,
                             int_t end = count,
                             comp_t greater = [](const key_t& a,
                                                 const key_t& b) {
                               return a > b;
                             })
  {
    while (begin < end) {
      int_t mid = floor((begin + end) / 2);
      key_t key_ = keys[mid];
      bool pred = greater(key_, key);
      if (pred) {
        end = mid;
      } else {
        begin = mid + 1;
      }
    }
    return end - 1;
  }

  } // namespace: block
  } // namespace: device
} // namespace: binary
} // namespace: search
} // namespace: algo
} // namespace: gunrock
