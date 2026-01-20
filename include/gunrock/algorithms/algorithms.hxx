/**
 * @file algorithms.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/framework/operators/configs.hxx>

namespace gunrock {

/**
 * @brief Common options struct for algorithm optimization settings.
 * 
 * This struct provides a unified way to configure operator behavior
 * across all algorithms. It can be embedded in algorithm-specific
 * param_t structs to allow runtime configuration of:
 * - Advance operator load balancing strategy
 * - Filter operator algorithm selection
 * - Uniquify operator settings
 */
struct options_t {
  /// Load balancing technique for the advance operator
  operators::load_balance_t advance_load_balance = 
      operators::load_balance_t::block_mapped;
  
  /// Algorithm to use for filter operator
  operators::filter_algorithm_t filter_algorithm = 
      operators::filter_algorithm_t::predicated;
  
  /// Whether to enable filter operator (algorithm-specific)
  bool enable_filter = false;
  
  /// Whether to enable uniquify operator (algorithm-specific)
  bool enable_uniquify = false;
  
  /// Algorithm to use for uniquify operator
  operators::uniquify_algorithm_t uniquify_algorithm = 
      operators::uniquify_algorithm_t::unique;
  
  /// Best-effort uniquification (skip sorting)
  bool best_effort_uniquify = true;
  
  /// Percentage of elements to uniquify (0-100)
  float uniquify_percent = 100.0f;
  
  /// Default constructor with sensible defaults
  options_t() = default;
  
  /// Constructor with all options
  options_t(operators::load_balance_t _advance_load_balance,
            operators::filter_algorithm_t _filter_algorithm = 
                operators::filter_algorithm_t::predicated,
            bool _enable_filter = false,
            bool _enable_uniquify = false,
            operators::uniquify_algorithm_t _uniquify_algorithm = 
                operators::uniquify_algorithm_t::unique,
            bool _best_effort_uniquify = true,
            float _uniquify_percent = 100.0f)
      : advance_load_balance(_advance_load_balance),
        filter_algorithm(_filter_algorithm),
        enable_filter(_enable_filter),
        enable_uniquify(_enable_uniquify),
        uniquify_algorithm(_uniquify_algorithm),
        best_effort_uniquify(_best_effort_uniquify),
        uniquify_percent(_uniquify_percent) {}
};

}  // namespace gunrock

// Core includes
#include <gunrock/memory.hxx>
#include <gunrock/error.hxx>

// Framework includes
#include <gunrock/framework/framework.hxx>

// Utility includes
#include <gunrock/util/math.hxx>
#include <gunrock/util/print.hxx>
#include <gunrock/util/compare.hxx>

// Format includes
#include <gunrock/formats/formats.hxx>

// I/O includes
#include <gunrock/io/matrix_market.hxx>
#include <gunrock/io/smtx.hxx>
#include <gunrock/io/sample.hxx>

// Graph includes
#include <gunrock/graph/graph.hxx>

// Container includes
#include <gunrock/container/array.hxx>
#include <gunrock/container/vector.hxx>