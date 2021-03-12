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

#include <gunrock/framework/frontier/frontier.hxx>
#include <gunrock/framework/problem.hxx>

#pragma once

namespace gunrock {

/**
 * @brief
 *
 */
struct enactor_properties_t {
  float frontier_sizing_factor{1.0};
  std::size_t number_of_frontier_buffers{2};
  enactor_properties_t() = default;
};

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
template <typename algorithm_problem_t,
          frontier_kind_t frontier_kind = frontier_kind_t::vertex_frontier>
struct enactor_t {
  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t = typename algorithm_problem_t::edge_t;

  using frontier_type = frontier_t<
      std::conditional_t<frontier_kind == frontier_kind_t::vertex_frontier,
                         vertex_t,
                         edge_t>>;

  enactor_properties_t properties;
  std::shared_ptr<cuda::multi_context_t> context;
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
            std::shared_ptr<cuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : problem(_problem),
        properties(_properties),
        context(_context),
        frontiers(properties.number_of_frontier_buffers),
        active_frontier(&frontiers[0]),
        inactive_frontier(&frontiers[1]),
        buffer_selector(0),
        iteration(0),
        scanned_work_domain(problem->get_graph().get_number_of_vertices()) {
    // Set temporary buffer to be at least the number of edges
    auto g = problem->get_graph();
    std::size_t initial_size =
        (g.get_number_of_edges() > g.get_number_of_vertices())
            ? g.get_number_of_edges()
            : g.get_number_of_vertices();

    for (auto& buffers : frontiers) {
      buffers.reserve(
          (std::size_t)(properties.frontier_sizing_factor * initial_size));
    }
  }

  /**
   * @brief Get the problem pointer object
   * @return algorithm_problem_t*
   */
  algorithm_problem_t* get_problem() { return problem; }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_type*
   */
  frontier_type* get_input_frontier() { return active_frontier; }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_type*
   */
  frontier_type* get_output_frontier() { return inactive_frontier; }

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
    auto context0 = context->get_context(0);
    prepare_frontier(get_input_frontier(), *context);
    auto timer = context0->timer();
    timer.begin();
    while (!is_converged(*context)) {
      loop(*context);
      ++iteration;
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
  virtual void loop(cuda::multi_context_t& context) = 0;

  /**
   * @brief Prepare the initial frontier.
   *
   * @param context
   */
  virtual void prepare_frontier(frontier_type* f,
                                cuda::multi_context_t& context) = 0;

  /**
   * @brief Algorithm is converged if true is returned, keep on iterating if
   * false is returned. This function is checked at the end of every iteration
   * of the enact().
   *
   * @return true
   * @return false
   */
  virtual bool is_converged(cuda::multi_context_t& context) {
    return active_frontier->is_empty();
  }

};  // struct enactor_t

}  // namespace gunrock