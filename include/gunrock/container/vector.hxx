/**
 * @file vector.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief support for stl based vectors on gpu. Relies on thrust::host and
 * device vectors.
 *
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/memory.hxx>
#include <gunrock/util/type_traits.hxx>

// includes: thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace gunrock {

template <typename type_t, memory::memory_space_t space>
struct vector {
  using type =
      std::conditional_t<space == memory::memory_space_t::host,  // condition
                         thrust::host_vector<type_t>,            // host_type
                         thrust::device_vector<type_t>           // device_type
                         >;
};

}  // namespace gunrock