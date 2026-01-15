/**
 * @file operators.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace operators {}  // namespace operators
}  // namespace gunrock

#include <gunrock/framework/operators/configs.hxx>

#include <gunrock/framework/operators/advance/advance.hxx>
#include <gunrock/framework/operators/filter/filter.hxx>
#include <gunrock/framework/operators/for/for.hxx>
#include <gunrock/framework/operators/uniquify/uniquify.hxx>
#include <gunrock/framework/operators/batch/batch.hxx>

#if __HIP_PLATFORM_NVIDIA__
#include <gunrock/framework/operators/neighborreduce/neighborreduce.hxx>
#endif