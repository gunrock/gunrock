/**
 * @file color.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Graph Coloring algorithm.
 * @date 2020-11-24
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/generate/random.hxx>

namespace gunrock {
namespace color {

struct param_t {
  options_t options;  ///< Optimization options (advance load-balance, filter, uniquify)
  
  param_t(options_t _options = options_t())
      : options(_options) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* colors;
  result_t(vertex_t* colors_) : colors(colors_) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<float> randoms;

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    // Allocate space for randoms array.
    randoms.resize(n_vertices);
  }

  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    auto d_colors = thrust::device_pointer_cast(this->result.colors);
    thrust::fill(thrust::device, d_colors + 0, d_colors + n_vertices,
                 gunrock::numeric_limits<vertex_t>::invalid());

    // Generate random numbers.
    generate::random::uniform_distribution(randoms, float(0.0f),
                                           float(n_vertices));
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto n_vertices = P->get_graph().get_number_of_vertices();

    // Fill the frontier with a sequence of vertices from 0 -> n_vertices.
    f->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto colors = P->result.colors;
    auto randoms = P->randoms.data().get();
    auto iteration = E->iteration;

    auto color_me_in = [G, colors, randoms, iteration] __host__ __device__(
                           vertex_t const& vertex) -> bool {
      edge_t num_neighbors = G.get_number_of_neighbors(vertex);

      // Color two nodes at the same time.
      const int color = iteration * 2;

      // Exit early if the vertex has no neighbors.
      if (num_neighbors == 0) {
        colors[vertex] = color;
        return false;  // remove (colored)
      }

      bool colormax = true;
      bool colormin = true;

      edge_t start_edge = G.get_starting_edge(vertex);
      auto rand_v = randoms[vertex];

      // Main loop that goes over all the neighbors and finds the maximum or
      // minimum random number vertex.
      for (edge_t e = start_edge; e < start_edge + num_neighbors; ++e) {
        vertex_t u = G.get_destination_vertex(e);

        if (gunrock::util::limits::is_valid(colors[u]) &&
                (colors[u] != color) && (colors[u] != color + 1) ||
            (vertex == u))
          continue;

        auto rand_u = randoms[u];
        if (rand_v < rand_u || (rand_v == rand_u && vertex < u))
          colormax = false;
        if (rand_v > rand_u || (rand_v == rand_u && vertex > u))
          colormin = false;
      }

      // Color if the node has the maximum OR minimum random number, this way,
      // per iteration we can possibly fill 2 colors at the same time.
      if (colormax) {
        colors[vertex] = color;
        return false;  // remove (colored).
      } else if (colormin) {
        colors[vertex] = color + 1;
        return false;  // remove (colored).
      } else {
        return true;  // keep (not colored).
      }
    };

    // Execute filter operator on the provided lambda using runtime dispatch
    auto filter_algorithm = P->param.options.filter_algorithm;
    operators::filter::execute_runtime(G, E, color_me_in, filter_algorithm, context);
  }

};  // struct enactor_t

/**
 * @brief Run Graph Coloring algorithm on a given graph, G, with provided
 * parameters and results.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param param Algorithm parameters (param_t).
 * @param result Algorithm results (result_t).
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          param_t& param,
          result_t<typename graph_t::vertex_type>& result,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t;
  using result_type = result_t<vertex_t>;

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

/**
 * @brief Run Graph Coloring algorithm on a given graph with simplified parameters.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param colors Output array of color assignments for each vertex.
 * @param filter_algorithm Filter algorithm to use (default: predicated).
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type* colors,  // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;

  param_t param;
  result_t<vertex_t> result(colors);

  return run(G, param, result, context);
}

}  // namespace color
}  // namespace gunrock
