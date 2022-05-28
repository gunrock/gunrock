/**
 * @file sssp.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path algorithm.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace sssp {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  vertex_t* predecessors;
  result_t(weight_t* _distances, vertex_t* _predecessors, vertex_t n_vertices)
      : distances(_distances), predecessors(_predecessors) {}
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

  thrust::device_vector<vertex_t> visited;

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    visited.resize(n_vertices);

    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill(policy, visited.begin(), visited.end(), -1);
  }

  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    auto context = this->get_single_context();
    auto policy = context->execution_policy();

    auto single_source = this->param.single_source;
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(policy, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<weight_t>::max());

    thrust::fill(policy, d_distances + single_source,
                 d_distances + single_source + 1, 0);

    thrust::fill(policy, visited.begin(), visited.end(),
                 -1);  // This does need to be reset in between runs though
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
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;

    auto shortest_path = [distances, single_source] __host__ __device__(
                             vertex_t const& source,    // ... source
                             vertex_t const& neighbor,  // neighbor
                             edge_t const& edge,        // edge
                             weight_t const& weight     // weight (tuple).
                             ) -> bool {
      weight_t source_distance = thread::load(&distances[source]);
      weight_t distance_to_neighbor = source_distance + weight;

      // Check if the destination node has been claimed as someone's child
      weight_t recover_distance =
          math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

      return (distance_to_neighbor < recover_distance);
    };

    auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
                                      vertex_t const& vertex) -> bool {
      if (visited[vertex] == iteration)
        return false;

      visited[vertex] = iteration;
      /// @todo Confirm we do not need the following for bug
      /// https://github.com/gunrock/essentials/issues/9 anymore.
      // return G.get_number_of_neighbors(vertex) > 0;
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, shortest_path, context);

    // Execute filter operator on the provided lambda
    operators::filter::execute<operators::filter_algorithm_t::bypass>(
        G, E, remove_completed_paths, context);

    /// @brief Execute uniquify operator to deduplicate the frontier
    /// @note Not required.
    // // bool best_effort_uniquification = true;
    // // operators::uniquify::execute<operators::uniquify_algorithm_t::unique>(
    // // E, context, best_effort_uniquification);
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::weight_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  param_type param(single_source);
  result_type result(distances, predecessors, G.get_number_of_vertices());
  // </user-defined>

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace sssp
}  // namespace gunrock