/**
 * @file sssp_implementation.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm. This is where
 * we actually implement SSSP using operators.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/applications/application.hxx>

namespace gunrock {
namespace sssp {

template <typename graph_type>
struct sssp_problem_t : problem_t<graph_type> {
  // Get useful types from graph_type
  using vertex_t = typename graph_type::vertex_type;
  using weight_pointer_t = typename graph_type::weight_pointer_t;
  using vertex_pointer_t = typename graph_type::vertex_pointer_t;

  // Useful types from problem_t
  using problem_type = problem_t<graph_type>;

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
  sssp_problem_t(graph_type* _G,
                 vertex_t& _source,
                 weight_pointer_t _distances,
                 vertex_pointer_t _predecessors)
      : problem_type(_G),
        single_source(_source),
        distances(_distances),
        predecessors(_predecessors) {}

  sssp_problem_t(const sssp_problem_t& rhs) = delete;
  sssp_problem_t& operator=(const sssp_problem_t& rhs) = delete;
};

template <typename algorithm_problem_t>
struct sssp_enactor_t : enactor_t<algorithm_problem_t> {
  using enactor_type = enactor_t<algorithm_problem_t>;

  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t = typename algorithm_problem_t::edge_t;
  using weight_t = typename algorithm_problem_t::weight_t;

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
  void loop(cuda::standard_context_t& context) {
    // Data slice
    auto P = enactor_type::get_problem_pointer();
    auto G = P->get_graph_pointer();
    auto distances = P->distances;
    auto single_source = P->single_source;

    std::cout << "Single source: " << single_source << std::endl;

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
          math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

      if (distance_to_neighbor < recover_distance)
        return true;
      // frontier::mark_to_keep(source);
      return false;
      // frontier::mark_for_removal(source);
    };

    /**
     * @brief Lambda operator to determine which vertices to filter and which
     * to keep.
     *
     */
    // auto remove_completed_paths =
    //     [] __host__ __device__(vertex_t const& vertex) -> bool {
    //   // if (!frontier::marked_for_removal(vertex))
    //   //   frontier::remove_from_frontier(vertex);
    //   // frontier::keep_in_frontier(vertex);
    //   return false;
    // };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::advance_type_t::vertex_to_vertex>(
        G, enactor_type::frontiers.data(), shortest_path);

    // // Execute filter operator on the provided lambda
    // operators::filter::execute(G, frontiers, remove_completed_paths);
  }

  void prepare_frontier(cuda::standard_context_t& context) {
    auto P = enactor_type::get_problem_pointer();
    auto single_source = P->single_source;
    auto G = P->get_graph_pointer();

    std::cout << G->memory_space() << std::endl;

    thrust::device_vector<vertex_t> frontier_data(1);
    frontier_data.push_back(single_source);
    auto active_buffer = enactor_type::get_active_frontier_buffer();
    active_buffer->load(frontier_data);
  }

  sssp_enactor_t(algorithm_problem_t* problem, cuda::multi_context_t& context)
      : enactor_type(problem, context) {}

  sssp_enactor_t(const sssp_enactor_t& rhs) = delete;
  sssp_enactor_t& operator=(const sssp_enactor_t& rhs) = delete;
};  // struct sssp_enactor_t

}  // namespace sssp
}  // namespace gunrock