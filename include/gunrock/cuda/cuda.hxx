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
namespace gcuda {}  // namespace gcuda
}  // namespace gunrock

#include <gunrock/cuda/global.hxx>
#include <gunrock/cuda/device.hxx>
#include <gunrock/cuda/function.hxx>
#include <gunrock/cuda/stream_management.hxx>
#include <gunrock/cuda/event_management.hxx>
#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/cuda/context.hxx>
#include <gunrock/cuda/sm.hxx>
#include <gunrock/cuda/launch_box.hxx>