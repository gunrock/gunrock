/**
 * @file enactor.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <vector>

#include <gunrock/framework/frontier.hxx>
#include <gunrock/framework/problem.hxx>

#pragma once

namespace gunrock {

template <typename problem_type>
struct enactor_t {
  using vertex_t = typename problem_type::graph_type::vertex_type;

  context_t context;
  std::shared_ptr<problem_type> problem;

  std::vector<std::shared_ptr<frontier_t<vertex_t>>> frontiers;

  // Disable copy ctor and assignment operator.
  // We don't want to let the user copy only a slice.
  enactor_t(const enactor_t& rhs) = delete;
  enactor_t& operator=(const enactor_t& rhs) = delete;

  enactor_t(std::shared_ptr<problem_type> problem, context_t& context)
      : problem(problem), context(context) {}

  /**
   * @brief Run the enactor with the given problem and the loop.
   *
   * @note We can work on evolving this into a multi-gpu implementation.
   *
   * @tparam algorithm_problem_t
   * @param problem
   * @param context
   * @return float
   */
  float enact(std::shared_ptr<problem_type> problem, context_t& context) {
    while (!is_converged()) {
      loop(problem, context);
    }
  }

  /**
   * @brief This is the core of the implementation for any algorithm. loops
   * till the convergence condition is met (see: is_converged()). Note that this
   * function is on the host and is timed, so make sure you are writing the most
   * efficient implementation possible. Avoid performing copies in this function
   * or running API calls that are incredibly slow (such as printfs), unless
   * they are part of your algorithms' implementation.
   *
   * @param context
   */
  virtual void loop(context_t& context) = 0;

  /**
   * @brief Algorithm is converged if true is returned, keep on iterating if
   * false is returned. This function is checked at the end of every iteration
   * of the enact().
   *
   * @return true
   * @return false
   */
  virtual bool is_converged(context_t& context) { return frontier.empty(); }

};  // struct enactor_t

}  // namespace gunrock