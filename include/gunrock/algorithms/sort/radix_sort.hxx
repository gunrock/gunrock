/**
 * @file radix_sort.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
#include <gunrock/cuda/stream_management.hxx>
#include <thrust/sort.h>

namespace gunrock {

/**
 * @namespace sort
 * Namespace for sort algorithm, includes sort on host and device. Algorithms
 * such as radix sort and stable sort (std::sort uses stable sort). We also
 * support segmented sort.
 */
namespace sort {

enum order_t { ascending, descending };

/**
 * @namespace radix
 * Namespace for radix sort algorithms on host and device. Supports sorting
 * keys, key-value pairs and segments.
 */
namespace radix {

// keys-only
// Sorts keys into ascending order. (~2N auxiliary storage required)
// DoubleBuffer: Sorts keys pairs into ascending order. (~N auxiliary storage
// required)

template <typename type_t>
void sort_keys(type_t* keys,
               std::size_t num_items,
               order_t order = order_t::ascending,
               gcuda::stream_t stream = 0) {
  if (order == order_t::ascending)
    thrust::sort(thrust::cuda::par.on(stream), keys, keys + num_items,
                 thrust::less<type_t>());
  else
    thrust::sort(thrust::cuda::par.on(stream), keys, keys + num_items,
                 thrust::greater<type_t>());
}

}  // namespace radix

}  // namespace sort
}  // namespace gunrock