/**
 * @file color_implementation.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm. This is where
 * we actually implement color using operators.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <bits/stdc++.h>
#include <cstdlib>

#include <gunrock/applications/application.hxx>

#include <gunrock/algorithms/generate/random.hxx>

namespace gunrock {
namespace color {

template <typename graph_type, typename host_graph_type>
struct color_problem_t : problem_t<graph_type, host_graph_type> {
  // Get useful types from graph_type
  using vertex_t = typename graph_type::vertex_type;
  using weight_t = typename graph_type::weight_type;

  using weight_pointer_t = typename graph_type::weight_pointer_t;
  using vertex_pointer_t = typename graph_type::vertex_pointer_t;

  // Useful types from problem_t
  using problem_type = problem_t<graph_type, host_graph_type>;

  thrust::device_vector<vertex_t> randoms;
  vertex_pointer_t colors;

  /**
   * @brief Construct a new color problem t object
   *
   * @param G graph on GPU
   * @param g graph on CPU
   * @param context system context
   * @param _colors output color per vertex array
   */
  color_problem_t(graph_type* G,
                  host_graph_type* g,
                  std::shared_ptr<cuda::multi_context_t> context,
                  vertex_pointer_t _colors)
      : problem_type(G, g, context),
        colors(_colors),
        randoms(g->get_number_of_vertices()) {
    // XXX: Ugly. Initialize d_colors to be all INVALIDs.
    auto d_colors = thrust::device_pointer_cast(colors);
    thrust::fill(thrust::device, d_colors + 0,
                 d_colors + g->get_number_of_vertices(),
                 std::numeric_limits<vertex_t>::max());

    // Generate random numbers.
    algo::generate::random::uniform_distribution(0, g->get_number_of_vertices(),
                                                 randoms.begin());
  }

  color_problem_t(const color_problem_t& rhs) = delete;
  color_problem_t& operator=(const color_problem_t& rhs) = delete;
};

template <typename algorithm_problem_t>
struct color_enactor_t : enactor_t<algorithm_problem_t> {
  using enactor_type = enactor_t<algorithm_problem_t>;

  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t = typename algorithm_problem_t::edge_t;
  using weight_t = typename algorithm_problem_t::weight_t;

  /**
   * @brief ... XXX
   *
   * @param context
   */
  void loop(cuda::standard_context_t* context) override {
    // Data slice
    auto E = enactor_type::get_enactor();
    auto P = E->get_problem_pointer();
    auto G = P->get_graph_pointer();

    auto colors = P->colors;
    auto rand = P->randoms.data().get();
    auto iteration = E->iteration;

    /**
     * @brief ... XXX
     *
     */
    auto color_me_in = [G, colors, rand, iteration] __host__ __device__(
                           vertex_t const& vertex) -> bool {
      // If invalid vertex, exit early.
      if (vertex == std::numeric_limits<vertex_t>::max())
        return false;

      edge_t start_edge = G->get_starting_edge(vertex);
      edge_t num_neighbors = G->get_number_of_neighbors(vertex);

      bool colormax = true;
      bool colormin = true;

      // Color two nodes at the same time.
      int color = iteration * 2;

      // Main loop that goes over all the neighbors and finds the maximum or
      // minimum random number vertex.
      for (edge_t e = start_edge; e < start_edge + num_neighbors; e++) {
        vertex_t u = G->get_destination_vertex(e);

        if ((colors[u] == std::numeric_limits<vertex_t>::max()) &&
                (colors[u] != color + 1) && (colors[u] != color + 2) ||
            (vertex == u))
          continue;
        if (rand[vertex] <= rand[u])
          colormax = false;
        if (rand[vertex] >= rand[u])
          colormin = false;
      }

      // Color if the node has the maximum OR minimum random number, this way,
      // per iteration we can possibly fill 2 colors at the same time.
      if (colormax) {
        colors[vertex] = color + 1;
        return false;  // remove (colored).
      } else if (colormin) {
        colors[vertex] = color + 2;
        return false;  // remove (colored).
      } else {
        return true;  // keep (not colored).
      }
    };

    // Execute filter operator on the provided lambda.
    operators::filter::execute<operators::filter_type_t::predicated>(
        G, E, color_me_in);
  }

  /**
   * @brief Populate the initial frontier with a the entire graph (nodes).
   *
   * @param context
   */
  void prepare_frontier(cuda::standard_context_t* context) override {
    auto E = enactor_type::get_enactor();      // Enactor pointer
    auto P = E->get_problem_pointer();         // Problem pointer
    auto g = P->get_host_graph_pointer();      // HOST graph pointer
    auto f = E->get_active_frontier_buffer();  // active frontier

    // XXX: Find a better way to initialize the frontier to all nodes
    for (vertex_t v = 0; v < g->get_number_of_vertices(); ++v)
      f->push_back(v);
  }

  color_enactor_t(algorithm_problem_t* problem,
                  std::shared_ptr<cuda::multi_context_t> context)
      : enactor_type(problem, context) {}

  color_enactor_t(const color_enactor_t& rhs) = delete;
  color_enactor_t& operator=(const color_enactor_t& rhs) = delete;
};  // struct color_enactor_t

}  // namespace color
}  // namespace gunrock