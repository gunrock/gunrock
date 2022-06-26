/**
 * @file mst.hxx
 * @author Annie Robison (amrobison@ucdavis.edu)
 * @brief Minimum Spanning Tree algorithm.
 * @version 0.1
 * @date 2022-03-17
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

using vertex_t = int;
using edge_t = int;
using weight_t = float;

namespace gunrock {
namespace mst {

template <typename vertex_t>
struct param_t {};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* mst_weight;
  result_t(weight_t* _mst_weight) : mst_weight(_mst_weight) {}
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

  graph_t g = this->get_graph();
  int n_vertices = g.get_number_of_vertices();

  thrust::device_vector<vertex_t> roots;
  thrust::device_vector<vertex_t> new_roots;
  thrust::device_vector<weight_t> min_weights;
  thrust::device_vector<edge_t> min_neighbors;
  thrust::device_vector<int> super_vertices;
  thrust::device_vector<bool> not_decremented;

  void init() {
    roots.resize(n_vertices);
    new_roots.resize(n_vertices);
    min_weights.resize(n_vertices);
    min_neighbors.resize(n_vertices);
    super_vertices.resize(1);
    not_decremented.resize(1);
  }

  void reset() {
    auto policy = this->context->get_context(0)->execution_policy();
    auto d_mst_weight = thrust::device_pointer_cast(this->result.mst_weight);

    thrust::fill(policy, min_weights.begin(), min_weights.end(),
                 std::numeric_limits<weight_t>::max());
    thrust::fill(policy, d_mst_weight, d_mst_weight + 1, 0);
    thrust::fill(policy, min_neighbors.begin(), min_neighbors.end(),
                 std::numeric_limits<edge_t>::max());
    thrust::fill(policy, super_vertices.begin(), super_vertices.end(),
                 n_vertices);
    thrust::fill(policy, not_decremented.begin(), not_decremented.end(), false);
    thrust::sequence(policy, roots.begin(), roots.end(), 0);
    thrust::sequence(policy, new_roots.begin(), new_roots.end(), 0);
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
    auto n_edges = P->get_graph().get_number_of_edges();

    // Fill the frontier with a sequence of edges from 0 -> n_edges
    f->sequence((edge_t)0, n_edges, context.get_context(0)->stream());
  }

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto mst_weight = P->result.mst_weight;
    auto super_vertices = P->super_vertices.data().get();
    auto min_neighbors = P->min_neighbors.data().get();
    auto roots = P->roots.data().get();
    auto new_roots = P->new_roots.data().get();
    auto min_weights = P->min_weights.data().get();
    auto policy = this->context->get_context(0)->execution_policy();
    auto not_decremented = P->not_decremented.data().get();
    // Reset on each iteration
    thrust::fill_n(policy, not_decremented, 1, true);
    thrust::fill_n(policy, min_weights, P->n_vertices,
                   std::numeric_limits<weight_t>::max());
    thrust::fill_n(policy, min_neighbors, P->n_vertices,
                   std::numeric_limits<weight_t>::max());

    auto get_min_weights = [min_weights, roots, G] __host__ __device__(
                               edge_t const& e  // id of edge
                               ) -> bool {
      // Find the minimum weight for each vertex. If the function returns true,
      // the weight is the current minimum and the edge will be added to the
      // output frontier.
      auto source = G.get_source_vertex(e);
      auto dest = G.get_destination_vertex(e);
      auto weight = G.get_edge_weight(e);

      // If the source and destination are already part of same super vertex, do
      // not check
      if (source < dest && roots[source] != roots[dest]) {
        auto old_weight1 =
            math::atomic::min(&(min_weights[roots[source]]), weight);
        auto old_weight2 =
            math::atomic::min(&(min_weights[roots[dest]]), weight);
        if (weight <= old_weight1 || weight <= old_weight2) {
          return true;
        }
      }
      return false;
    };

    auto get_min_neighbors =
        [G, min_weights, min_neighbors, roots] __host__ __device__(
            edge_t const& e  // id of edge
            ) -> void {
      // Find the minimum neighbor for each vertex. Use atomic min to break ties
      // between neighbors that have the same weight.
      // Consistent ordering (using min here) will prevent loops.
      // Edges with dest < source are flipped so that reverse edges are treated
      // as equivalent. Must check that the weight equals the min weight for
      // that vertex, because some edges can be added to the frontier that are
      // later beaten by lower weights.
      auto source = G.get_source_vertex(e);
      auto dest = G.get_destination_vertex(e);
      auto weight = G.get_edge_weight(e);

      if (source < dest && roots[source] != roots[dest]) {
        if (weight == min_weights[roots[source]]) {
          math::atomic::min(&(min_neighbors[roots[source]]), e);
        }
        if (weight == min_weights[roots[dest]]) {
          math::atomic::min(&(min_neighbors[roots[dest]]), e);
        }
      }
    };

    auto add_to_mst = [G, roots, mst_weight, super_vertices, min_neighbors,
                       min_weights, new_roots, not_decremented] __host__
                      __device__(vertex_t const& v) -> void {
      // Add weights to MST. To prevent duplicate edges, check that either
      // the source vertex index is
      // less than the destination vertex index or that the edge with the source
      // and destination flipped is not included.
      // Combine super vertices by setting the root of the super vertex to the
      // root of the destination. Use `roots` to check old roots and update
      // roots in `new_roots`.
      if (min_weights[v] != std::numeric_limits<weight_t>::max()) {
        auto e = min_neighbors[v];
        auto source = G.get_source_vertex(e);
        auto dest = G.get_destination_vertex(e);
        auto weight = G.get_edge_weight(e);

        if (roots[source] != v) {
          auto temp = source;
          source = dest;
          dest = temp;
        }

        if (source < dest || min_neighbors[roots[dest]] != e) {
          // For large graphs with float weights, there may be slight variance
          // in the final MST weight due to atomic adds and the amount of
          // precision loss depending on the order of adds.
          not_decremented[0] = false;
          math::atomic::add(&mst_weight[0], weight);
          math::atomic::add(&super_vertices[0], -1);
          math::atomic::exch(&new_roots[v], new_roots[dest]);
        }
      }
    };

    auto jump_pointers_parallel =
        [new_roots] __host__ __device__(vertex_t const& v) -> void {
      // Update the root of each vertex. When adding an edge to the MST, we
      // update the source's root's root to the destination's root.
      // However, any vertex that had the source's root as its root would not
      // be updated. We must jump from root to root until the current vertex
      // and root are equal to find the new roots.
      vertex_t u = new_roots[v];
      while (new_roots[u] != u) {
        u = new_roots[u];
      }
      new_roots[v] = u;
      return;
    };

    auto in_frontier = &(this->frontiers[0]);
    auto out_frontier = &(this->frontiers[1]);

    // Execute filter operator to get min weights
    operators::filter::execute<operators::filter_algorithm_t::remove>(
        G, get_min_weights, in_frontier, out_frontier, context);

    // Execute parallel for operator to get min neighbors
    operators::parallel_for::execute<operators::parallel_for_each_t::element>(
        *out_frontier, get_min_neighbors, context);

    // Execute parallel for operator to add weights to MST
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,           // graph
        add_to_mst,  // lambda function
        context      // context
    );

    // Throw an exception if the number of super vertices has not been
    // decremented
    thrust::host_vector<bool> h_not_dec = P->not_decremented;
    error::throw_if_exception(
        h_not_dec[0],
        "Error: invalid graph (super vertices not decremented)\n");

    // Execute parallel for to jump pointers
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,                       // graph
        jump_pointers_parallel,  // lambda function
        context                  // context
    );

    // Copy `new_roots` to `roots`
    thrust::copy_n(policy, new_roots, P->n_vertices, roots);
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    return (P->super_vertices[0] == 1);
  }
};

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type* mst_weight,  // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  param_type param;
  result_type result(mst_weight);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace mst
}  // namespace gunrock
