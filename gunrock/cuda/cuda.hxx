/**
 * @file cuda.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace cuda {
typedef cudaStream_t stream_t;

}  // namespace cuda
}  // namespace gunrock

#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/cuda/context.hxx>