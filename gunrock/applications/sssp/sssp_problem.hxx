/**
 * @file sssp_problem.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Set-up data for Single-Source Shortest Path graph algorithm.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/framework/problem.hxx>

#pragma once

namespace gunrock {
namespace sssp {

template <typename graph_type>
struct sssp_problem_t : problem_t<graph_type> {
  typename graph_type::vertex_type single_source;

  typename graph_type::weight_pointer_t distances;
  typename graph_type::vertex_pointer_t predecessors;

  sssp_problem_t() {}

  sssp_problem_t(const sssp_problem_t& rhs) = delete;
  sssp_problem_t& operator=(const sssp_problem_t& rhs) = delete;
};

}  // namespace sssp
}  // namespace gunrock