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

#include <gunrock/cuda/cuda.hxx>
#include <gunrock/util/timer.hxx>

#include <gunrock/framework/frontier.hxx>
#include <gunrock/framework/problem.hxx>

#pragma once

namespace gunrock {

/**
 * @brief Building block of the algorithm within gunrock. An enactor structure
 * defines how, what and wehre the problem is going to be executed. Note that
 * the enactor enact() function can be extended to support multi-gpu contexts
 * (using execution policy like model). Enactor also has two pure virtual
 * functions, which MUST be implemented within the algorithm the user is trying
 * to write the enactor for. These functions prepare the initial frontier of the
 * algorithm (which could be one node, the entire graph or any variation), and
 * finally defines the main loop of iteration that iterates until the algorithm
 * converges. Default convergence condition is when the frontier is empty, the
 * algorithm has finished, but this convergence function can ALSO be extended to
 * support any custom convergence condition.
 *
 * @tparam algorithm_problem_t algorithm specific problem type
 */
template <typename algorithm_problem_t>
struct enactor_t {
  using vertex_t = typename algorithm_problem_t::vertex_t;
  using frontier_type = frontier_t<vertex_t>;

  static constexpr std::size_t number_of_buffers = 2;
  std::shared_ptr<cuda::multi_context_t> context;
  util::timer_t timer;  // XXX: needs to be a vector to support multi-gpu timer
                        // or we can move this within the actual context.
  algorithm_problem_t* problem;
  thrust::host_vector<frontier_type> frontiers;
  thrust::device_vector<vertex_t> scanned_work_domain;
  frontier_type* active_frontier;
  frontier_type* inactive_frontier;
  int buffer_selector;
  int iteration;

  // Disable copy ctor and assignment operator.
  // We don't want to let the user copy only a slice.
  enactor_t(const enactor_t& rhs) = delete;
  enactor_t& operator=(const enactor_t& rhs) = delete;

  enactor_t(algorithm_problem_t* _problem,
            std::shared_ptr<cuda::multi_context_t> _context)
      : problem(_problem),
        context(_context),
        frontiers(number_of_buffers),
        active_frontier(&frontiers[0]),
        inactive_frontier(&frontiers[1]),
        buffer_selector(0),
        iteration(0),
        scanned_work_domain(
            problem->get_host_graph_pointer()->get_number_of_vertices()) {
    // Set temporary buffer to be at least the number of edges
    auto g = problem->get_host_graph_pointer();
    auto buffer = get_inactive_frontier_buffer();
    buffer->reserve(g->get_number_of_edges());
  }

  /**
   * @brief Get the problem pointer object
   * @return algorithm_problem_t*
   */
  algorithm_problem_t* get_problem_pointer() { return problem; }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_type*
   */
  frontier_type* get_active_frontier_buffer() { return active_frontier; }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_type*
   */
  frontier_type* get_inactive_frontier_buffer() { return inactive_frontier; }

  void swap_frontier_buffers() {
    buffer_selector ^= 1;
    active_frontier = &frontiers[buffer_selector];
    inactive_frontier = &frontiers[buffer_selector ^ 1];
  }

  enactor_t* get_enactor() { return this; }

  /**
   * @brief Run the enactor with the given problem and the loop.
   * @note We can work on evolving this into a multi-gpu implementation.
   * @return float time took for enactor to complete.
   */
  float enact() {
    auto single_context = context->get_context(0);
    single_context->print_properties();
    prepare_frontier(single_context);
    timer.begin();
    while (!is_converged(single_context)) {
      loop(single_context);
      iteration++;
    }
    return timer.end();
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
  virtual void loop(cuda::standard_context_t* context) = 0;

  /**
   * @brief Prepare the initial frontier.
   *
   * @param context
   */
  virtual void prepare_frontier(cuda::standard_context_t* context) = 0;

  /**
   * @brief Algorithm is converged if true is returned, keep on iterating if
   * false is returned. This function is checked at the end of every iteration
   * of the enact().
   *
   * @return true
   * @return false
   */
  virtual bool is_converged(cuda::standard_context_t* context) {
    return active_frontier->empty();
  }

};  // struct enactor_t

}  // namespace gunrock