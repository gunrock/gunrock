/**
 * @file sssp.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path algorithm.
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
  options_t options;  ///< Optimization options (advance load-balance, filter, uniquify)
  
  param_t(vertex_t _single_source, options_t _options = options_t()) 
    : single_source(_single_source), options(_options) {}
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
    auto advance_load_balance = P->param.options.advance_load_balance;
    operators::advance::execute_runtime(G, E, shortest_path, advance_load_balance, context);

    // Execute filter operator on the provided lambda
    // SSSP uses bypass filter for visited vertices tracking
    operators::filter::execute<operators::filter_algorithm_t::bypass>(
        G, E, remove_completed_paths, context);

    // Execute uniquify operator to deduplicate the frontier (if enabled via options)
    if (P->param.options.enable_uniquify) {
      operators::uniquify::execute<operators::uniquify_algorithm_t::unique>(
          E, context, P->param.options.best_effort_uniquify,
          P->param.options.uniquify_percent);
    }
  }

};  // struct enactor_t

/**
 * @brief Run Single-Source Shortest Path algorithm on a given graph, G, with
 * provided parameters and results.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param param Algorithm parameters (param_t) including source and options.
 * @param result Algorithm results (result_t) with output pointers.
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          param_t<typename graph_t::vertex_type>& param,
          result_t<typename graph_t::vertex_type, typename graph_t::weight_type>& result,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  // Create problem and enactor in a scope to ensure proper cleanup
  float runtime = 0.0f;
  {
    problem_type problem(G, param, result, context);
    problem.init();
    problem.reset();
    
    // Synchronize after reset to ensure initialization completes
    context->get_context(0)->synchronize();

    enactor_type enactor(&problem, context);
    runtime = enactor.enact();
    
    // Synchronize context to ensure all GPU operations complete
    // before problem/enactor destructors run
    context->get_context(0)->synchronize();
  }
  // Problem and enactor are now fully destroyed
  
  // Final device synchronization to ensure all operations are complete
  // before the next run starts
  auto single_context = context->get_context(0);
  single_context->synchronize();
  
  return runtime;
}

/**
 * @brief Run Single-Source Shortest Path algorithm on a given graph, G,
 * starting from the source node, single_source.
 *
 * @note This is a legacy API that delegates to the new param/result API.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::weight_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  param_t<vertex_t> param(single_source);
  result_t<vertex_t, weight_t> result(distances, predecessors, G.get_number_of_vertices());

  return run(G, param, result, context);
}

}  // namespace sssp
}  // namespace gunrock
