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

#include <gunrock/framework/framework.hxx>

#pragma once

namespace gunrock {
namespace sssp {

template <typename graph_type>
struct sssp_problem_t : problem_t<graph_type> {
  // Get useful types from graph_type
  using vertex_t = typename graph_type::vertex_type;
  using weight_pointer_t = typename graph_type::weight_pointer_t;
  using vertex_pointer_t = typename graph_type::vertex_pointer_t;

  vertex_t single_source;
  weight_pointer_t distances;
  vertex_pointer_t predecessors;

  /**
   * @brief Construct a new sssp problem t object
   *
   * @param _G  input graph
   * @param _source input single source for sssp
   * @param _distances output distance pointer
   * @param _predecessors output predecessors pointer
   */
  sssp_problem_t(graph_type& _G,
                 vertex_t& _source,
                 weight_pointer_t _distances,
                 vertex_pointer_t _predecessors)
      : problem_t<graph_type>(_G),
        single_source(_source),
        distances(_distances),
        predecessors(_predecessors) {}

  sssp_problem_t(const sssp_problem_t& rhs) = delete;
  sssp_problem_t& operator=(const sssp_problem_t& rhs) = delete;
};

}  // namespace sssp
}  // namespace gunrock