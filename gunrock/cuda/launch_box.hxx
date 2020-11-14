/**
 * @file device_properties.hxx
 * @author Cameron Shinn (ctshinn@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-11-09
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/cuda/device_properties.hxx>

namespace gunrock {
namespace cuda {

// Should the struct members be template parameters?
template<int blockDim_, int gridDim_, size_t smemBytes_ = 0>
struct launch_params_t{
  int blockDim = blockDim_;
  int gridDim = gridDim_;
  size_t smemBytes = smemBytes_;
};

// template<int ccCombined, int... launch_tparams>
// struct cc_launch_params_t : launch_params_t<launch_tparams...> {
//     // Compute capability related things
// };

// smemBytes_ isn't an int, but I am assuming it will get cast to a size_t? I think there is a better way to generalize the type
template<int... launch_tparams>
struct arch_launch_params_t : launch_params_t<launch_tparams...> {
  // Architecture related things
};

// Is there a better way to create arch types?
#define NAMED_ARCH_LP_TYPE(name) \
template<int... launch_tparams> \
using name = struct arch_launch_params_t<launch_tparams...>;

NAMED_ARCH_LP_TYPE(ampere)
NAMED_ARCH_LP_TYPE(kepler)
NAMED_ARCH_LP_TYPE(maxwell)
NAMED_ARCH_LP_TYPE(pascal)
NAMED_ARCH_LP_TYPE(volta)
NAMED_ARCH_LP_TYPE(turing)

template<typename... archLaunchParams>
struct launch_box_t {
  // Provide a way to get the correct params?
};

}  // namespace gunrock
}  // namespace cuda
