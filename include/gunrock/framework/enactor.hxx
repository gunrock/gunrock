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
 * @brief A simple struct to store enactor properties.
 *
 * @par Overview
 * Modify this per-algorithm if the defaults need to be changed. A
 * gunrock-specific parameter can also control the frontier sizing factor used
 * to resize and scale the frontier larger than the needed size.
 */
struct enactor_properties_t {
  /*!
   * Resizes the frontier by this factor * the size required.
   * @code frontier.resize(frontier_sizing_factor * new_size);
   */
  float frontier_sizing_factor{1.5f};

  /*!
   * Number of frontier buffers to manage.
   * @note Not used in the implementation yet.
   */
  std::size_t number_of_frontier_buffers{2};

  /*!
   * When enabled, enactor's internal frontiers are not allocated ahead of time.
   * Instead, the user is encouraged to manage the frontiers.
   */
  bool self_manage_frontiers{false};

  /**
   * @brief Construct a new enactor properties t object with default values.
   */
  enactor_properties_t() = default;
};

/**
 * @brief Building block of the algorithm within gunrock. The methods within the
 * enactor structure are overriden by an application writer (methods such as
 * loop) to extend the enactor to implement a graph algorithm.
 *
 * @par Overview
 * Note that the enactor's `enact()` member can be extended to support
 * multi-gpu contexts (using execution policies like model). Enactor also has
 * pure virtual functions, which **must** be implemented within the graph
 * algorithm by the developers. These functions prepare the initial frontier of
 * the algorithm (which could be one node, the entire graph or any subset),
 * and defines the main loop of iteration that iterates until the algorithm
 * converges.
 * @see prepare_frontier()
 * @see loop()
 * @see is_converged()
 *
 * @tparam algorithm_problem_t algorithm specific problem type.
 * @tparam frontier_kind `gunrock::frontier_kind_t` enum (vertex or
 * edge-based frontier). Currently, only vertex frontier is supported.
 *
 */
template <typename algorithm_problem_t,
          frontier::frontier_kind_t frontier_kind =
              frontier::frontier_kind_t::vertex_frontier,
          frontier::frontier_view_t frontier_view =
              frontier::frontier_view_t::vector>
struct enactor_t {
  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t = typename algorithm_problem_t::edge_t;

  using frontier_t = frontier::frontier_t<vertex_t, edge_t, frontier_kind>;

  /*!
   * Enactor properties (frontier resizing factor, buffers, etc.)
   */
  enactor_properties_t properties;

  /*!
   * A shared_ptr to a multi-gpu context.
   * @see `gunrock::gcuda::multi_context_t`
   */
  std::shared_ptr<gcuda::multi_context_t> context;

  /*!
   * Algorithm's problem structure.
   */
  algorithm_problem_t* problem;

  /*!
   * Vector of frontier buffers (default number of buffers = 2).
   */
  thrust::host_vector<frontier_t> frontiers;

  /*!
   * Extra space to store scan of the work domain (for advance).
   * @note This is not used for all advance algorithms.
   * @todo Find a way to only allocate this if the advance algorithm that
   * actually needs it is being run. Otherwise, it maybe a waste of memory space
   * to allocate this.
   */
  thrust::device_vector<edge_t> scanned_work_domain;

  /*!
   * Active frontier buffer, this pointer can be obtained by
   * `get_input_frontier()` method. This buffer is used as an input frontier.
   */
  frontier_t* active_frontier;

  /*!
   * Inactive frontier buffer, this pointer can be obtained by
   * `get_output_frontier()` method. This buffer is often but not always used as
   * an output frontier.
   */
  frontier_t* inactive_frontier;

  /*!
   * An internal selector used to swap frontier buffers.
   */
  int buffer_selector;

  /*!
   * Number of iterations for the `loop` method. Increments per loop-iteration.
   */
  int iteration;

  /*!
   * Disable copy ctor and assignment operator. We don't want to let the user
   * copy only a slice.
   * @todo Look at enabling copy constructors.
   */
  enactor_t(const enactor_t& rhs) = delete;
  enactor_t& operator=(const enactor_t& rhs) = delete;

  /**
   * @brief Construct a new enactor t object.
   *
   * @param _problem algorithm's problem data structure.
   * @param _context shared pointer to the gcuda::multi_context_t context that
   * stores information about multiple GPUs (such as streams, device ids,
   * events, etc.)
   * @param _properties `gunrock::enactor_properties_t`, includes
   * properties such as frontier resizing factor (default = 2) and number of
   * frontier buffers to create for the enactor.
   */
  enactor_t(algorithm_problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : properties(_properties),
        context(_context),
        problem(_problem),
        frontiers(properties.number_of_frontier_buffers),
        scanned_work_domain(problem->get_graph().get_number_of_vertices() + 1),
        active_frontier(reinterpret_cast<frontier_t*>(&frontiers[0])),
        inactive_frontier(reinterpret_cast<frontier_t*>(&frontiers[1])),
        buffer_selector(0),
        iteration(0) {
    /*!
     * If the self manage frontiers property is false, the enactor interface
     * will resize the frontier buffers ahead of time to avoid the first
     * allocation during the algorithm execution face.
     * @note We also use a resizing factor to allocate more than what we
     * actually may need.
     *
     */
    if (!(properties.self_manage_frontiers)) {
      auto g = problem->get_graph();
      std::size_t initial_size =
          (g.get_number_of_edges() > g.get_number_of_vertices())
              ? g.get_number_of_edges()
              : g.get_number_of_vertices();

      for (auto& buffer : frontiers) {
        buffer.set_resizing_factor(properties.frontier_sizing_factor);
        buffer.reserve((std::size_t)(initial_size));
      }
    }
  }

  /**
   * @brief Get the problem pointer object
   * @return algorithm_problem_t*
   */
  algorithm_problem_t* get_problem() {
    return reinterpret_cast<algorithm_problem_t*>(problem);
  }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_t*
   */
  frontier_t* get_input_frontier() {
    return reinterpret_cast<frontier_t*>(active_frontier);
  }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_t*
   */
  frontier_t* get_output_frontier() {
    return reinterpret_cast<frontier_t*>(inactive_frontier);
  }

  /**
   * @brief Get the pointer to the enactor object.
   * @return enactor_t*
   */
  enactor_t* get_enactor() { return this; }

  /**
   * @brief Swap the inactive and active frontier buffers such that the inactive
   * buffer becomes the input buffer to the next operator and vice-versa.
   */
  void swap_frontier_buffers() {
    buffer_selector ^= 1;
    active_frontier =
        reinterpret_cast<frontier_t*>(&frontiers[buffer_selector]);
    inactive_frontier =
        reinterpret_cast<frontier_t*>(&frontiers[buffer_selector ^ 1]);
  }

  /**
   * @brief Run the enactor with the given problem and the loop.
   * @todo We can work on evolving this into a multi-gpu implementation.
   * @return float time took for enactor to complete (this is often used as
   * **the** time for performance measurements).
   */
  float enact() {
    auto single_context = context->get_context(0);
    prepare_frontier(get_input_frontier(), *context);
    auto timer = single_context->timer();
    timer.begin();
    while (!is_converged(*context)) {
      loop(*context);
      ++iteration;
    }
    finalize(*context);
    return timer.end();
  }

  /**
   * @brief This is the core of the implementation for any algorithm. Graph
   * algorithm developers should override this virtual function to implement
   * their own `loop` function, which loops till a convergence condition is
   * met `is_converged()`.
   *
   * @par Overview
   * This function is on the host and is timed, so make sure you are writing the
   * most efficient implementation possible. Avoid performing copies in this
   * function if they are not part of the algorithm's core, or running API calls
   * that are incredibly slow (such as `printfs` or debug statements).
   *
   * @param context `gunrock::gcuda::multi_context_t`.
   */
  virtual void loop(gcuda::multi_context_t& context) = 0;

  /**
   * @brief Prepare the initial frontier.
   *
   * @param context `gunrock::gcuda::multi_context_t`.
   */
  virtual void prepare_frontier(frontier_t* f,
                                gcuda::multi_context_t& context){};

  /**
   * @brief Algorithm is converged if true is returned, keep on iterating if
   * false is returned. This function is checked at the end of every iteration
   * of the `enact()`. Graph algorithm developers may override this function
   * with their own custom convergence condition.
   *
   * @par Overview
   * Default behavior of `is_converged()` method is to claim that the algorithm
   * is converged if the frontier size is set to be empty (`frontier->is_empty()
   * == true`).
   *
   * @return true converged!
   * @return false not converged, keep looping!
   */
  virtual bool is_converged(gcuda::multi_context_t& context) {
    return active_frontier->is_empty();
  }

  /**
   * @brief `finalize()` runs only once, and after the `while()` has ended.
   *
   * @par Finalize runs after the convergence condition is met in
   * `is_converged()`. Users can extend the following virtual function to do a
   * one final wrap-up of the application. Users are not required to implement
   * this function.
   *
   * @param context `gunrock::gcuda::multi_context_t`.
   */
  virtual void finalize(gcuda::multi_context_t& context) {}

};  // struct enactor_t

}  // namespace gunrock