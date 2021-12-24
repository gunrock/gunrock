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

using namespace memory;

template <typename type_t, memory_space_t space = memory_space_t::device>
using vector_t =
    std::conditional_t<space == memory_space_t::host,  // condition
                       thrust::host_vector<type_t>,    // host_type
                       thrust::device_vector<type_t>   // device_type
                       >;

template <typename type_t>
using host_vector_t = thrust::host_vector<type_t>;
template <typename type_t>
using device_vector_t = thrust::device_vector<type_t>;

}  // namespace gunrock