/**
 * @file vector.cuh
 *
 * @brief
 *
 * @todo support for stl based vectors on gpu.
 *
 */

#pragma once

#include <gunrock/util/type_traits.hxx>
#include <gunrock/memory.hxx>

// includes: thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace gunrock {

template <typename type_t, memory::memory_space_t space>
struct vector {
    using type = std::conditional_t<
                    space == memory::memory_space_t::host, // condition 
                    thrust::host_vector<type_t>,           // host_type
                    thrust::device_vector<type_t>          // device_type
                >;
};

}   // namespace gunrock