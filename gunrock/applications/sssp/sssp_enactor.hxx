/**
 * @file sssp_problem.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Enactor for Single-Source Shortest Path graph algorithm. This is where
 * we actually implement SSSP using operators.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <gunrock/framework/enactor.hxx>

#pragma once

namespace gunrock {
namespace sssp {

template <typename graph_type>
struct sssp_enactor_t : enactor_t<graph_type> {
  /**
   * @brief This is the core of the implementation for SSSP algorithm. loops
   * till the convergence condition is met (see: is_converged()). Note that this
   * function is on the host and is timed, so make sure you are writing the most
   * efficient implementation possible. Avoid performing copies in this function
   * or running API calls that are incredibly slow (such as printfs), unless
   * they are part of your algorithms' implementation.
   *
   * @param context
   */
  template <typename algorithm_problem_t>
  void loop(std::shared_ptr<algorithm_problem_t> problem, context_t& context) {
    // Data slice
    auto distances = problem->distances;
    auto single_source = problem->single_source;

    /**
     * @brief Lambda operator to advance to neighboring vertices from the
     * source vertices in the frontier, and marking the vertex to stay in the
     * frontier if and only if it finds a new shortest distance, otherwise,
     * it's shortest distance is found and we mark to remove the vertex from
     * the frontier.
     *
     */
    auto shortest_path = [distances, single_source] __host__ __device__(
                             vertex_t const& source, vertex_t const& neighbor,
                             edge_t const& edge,
                             weight_t const& weight) -> bool {
      weight_t source_distance = distances[source];  // use cached::load
      weight_t distance_to_neighbor = source_distance + weight;

      // Check if the destination node has been claimed as someone's child
      weight_t recover_distance =
          math::min::atomic(distances[neighbor], distance_to_neighbor);

      if (distance_to_neighbor < recover_distance)
        frontier::mark_to_keep(source);

      frontier::mark_for_removal(source);
    };

    /**
     * @brief Lambda operator to determine which vertices to filter and which
     * to keep.
     *
     */
    auto remove_completed_paths =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      if (!frontier::marked_for_removal(vertex))
        frontier::remove_from_frontier(vertex);
      frontier::keep_in_frontier(vertex);
    };

    // Execute advance operator on the provided lambda
    operator ::advance::execute<operator ::advance_type::vertex_to_vertex>(
        G, frontier, shortest_path);

    // Execute filter operator on the provided lambda
    operator ::filter::execute(G, frontier, remove_completed_paths);
  }

  sssp_enactor_t(context_t& context) : enactor_t(context) {}

  sssp_enactor_t(const sssp_enactor_t& rhs) = delete;
  sssp_enactor_t& operator=(const sssp_enactor_t& rhs) = delete;
};  // struct sssp_enactor_t

}  // namespace sssp
}  // namespace gunrock