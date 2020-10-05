/**
 * @file algorithms.hxx
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
namespace algo {

// Forward declaration...

}  // namespace algo
}  // namespace gunrock

// Raking Algorithms
#include <gunrock/algorithms/rake/rake.hxx>

// Scan & Reduce Algorithms
#include <gunrock/algorithms/reduce/reduce.hxx>
#include <gunrock/algorithms/scan/scan.hxx>

// Search Algorithms
#include <gunrock/algorithms/search/binary_search.hxx>
#include <gunrock/algorithms/search/sorted_search.hxx>

// Sort Algorithms
#include <gunrock/algorithms/sort/radix_sort.hxx>
#include <gunrock/algorithms/sort/stable_sort.hxx>