/**
 * @file operators.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace operators {

/**
 * @brief Load balancing type options, not all operators support load-balancing
 * or have need to balance work.
 *
 */
enum load_balance_type {
  merge_based,    // Merrill & Garland (SpMV)
  bucketing,      // Davidson et al. (SSSP)
  work_stealing,  // <find cite>
  none            // No ;oad-balancing applied
};

}  // namespace operators
}  // namespace gunrock

#include <gunrock/framework/operators/advance/advance.hxx>
#include <gunrock/framework/operators/filter/filter.hxx>