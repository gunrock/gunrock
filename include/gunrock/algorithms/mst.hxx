#pragma once

#include <gunrock/algorithms/algorithms.hxx>

using vertex_t = int;
using edge_t = int;
using weight_t = float;

namespace gunrock {
namespace mst {

template <typename vertex_t>
struct param_t {
  // No parameters for this algorithm
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* mst_weight;
  result_t(weight_t* _mst_weight) : mst_weight(_mst_weight) {}
};

// <boilerplate>
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  // </boilerplate>
  graph_t g = this->get_graph();
  int n_vertices = g.get_number_of_vertices();

  thrust::device_vector<vertex_t> roots;
  thrust::device_vector<vertex_t> new_roots;
  thrust::device_vector<weight_t> min_weights;
  thrust::device_vector<int> mst_edges;
  thrust::device_vector<int> super_vertices;
  thrust::device_vector<int> min_neighbors;

  void init() {
    auto policy = this->context->get_context(0)->execution_policy();

    roots.resize(n_vertices);
    new_roots.resize(n_vertices);
    min_weights.resize(n_vertices);
    min_neighbors.resize(n_vertices);

    mst_edges.resize(1);
    super_vertices.resize(1);

    auto d_mst_weight = thrust::device_pointer_cast(this->result.mst_weight);
    thrust::fill(policy, min_weights.begin(), min_weights.end(),
                 std::numeric_limits<weight_t>::max());
    thrust::fill(policy, d_mst_weight, d_mst_weight + 1, 0);
    thrust::fill(policy, min_neighbors.begin(), min_neighbors.end(), -1);
    thrust::sequence(policy, roots.begin(), roots.end(), 0);
    thrust::sequence(policy, new_roots.begin(), new_roots.end(), 0);
    thrust::sequence(policy, super_vertices.begin(), super_vertices.end(),
                     n_vertices);
  }

  void reset() {
    // TODO: reset
    return;
  }
};

// <boilerplate>
template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;
  // </boilerplate>

  // How to initialize the frontier at the beginning of the application.
  void prepare_frontier(frontier_t* f,
                        cuda::multi_context_t& context) override {
    // get pointer to the problem
    auto P = this->get_problem();
    auto n_vertices = P->get_graph().get_number_of_vertices();

    // Fill the frontier with a sequence of vertices from 0 -> n_vertices.
    f->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());
  }

  // One iteration of the application
  void loop(cuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto mst_weight = P->result.mst_weight;
    auto mst_edges = P->mst_edges.data().get();
    auto super_vertices = P->super_vertices.data().get();
    auto min_neighbors = P->min_neighbors.data().get();
    auto roots = P->roots.data().get();
    auto new_roots = P->new_roots.data().get();
    auto min_weights = P->min_weights.data().get();

    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill_n(policy, min_weights, P->n_vertices,
                   std::numeric_limits<weight_t>::max());
    thrust::fill_n(policy, min_neighbors, P->n_vertices, -1);

    // Find minimum weight for each vertex
    // TODO: update for multi-directional edges?
    auto get_min_weights = [min_weights, roots, G] __host__ __device__(
                               edge_t const& e  // id of edge
                               ) -> void {
      auto source = G.get_source_vertex(e);
      auto neighbor = G.get_destination_vertex(e);
      auto weight = G.get_edge_weight(e);

      // If they are already part of same super vertex, do not check
      if (roots[source] != roots[neighbor]) {
        // Store minimum weight
        auto old_weight =
            math::atomic::min(&(min_weights[roots[source]]), weight);
        printf(
            "1: source %i roots[source] %i weight %f min weight %f weight < "
            "old weight %i\n",
            source, roots[source], weight, min_weights[roots[source]],
            weight < old_weight);
      }
    };

    // TODO: technically this is non-deterministic between edges that are tied
    auto get_min_neighbors =
        [G, min_weights, min_neighbors, roots] __host__ __device__(
            edge_t const& e  // id of edge
            ) -> void {
      auto source = G.get_source_vertex(e);
      auto neighbor = G.get_destination_vertex(e);
      auto weight = G.get_edge_weight(e);

      // Keep neighbor if it is the min
      if (weight == min_weights[roots[source]]) {
        atomicCAS(&min_neighbors[roots[source]], -1, e);
        printf("source %i root %i min_neighbor %i\n", source, roots[source],
               min_neighbors[roots[source]]);
      }
    };

    // Add weights to MST
    auto add_to_mst = [G, roots, mst_weight, mst_edges, super_vertices,
                       min_neighbors, min_weights, new_roots] __host__
                      __device__(vertex_t const& v) -> void {
      if (min_weights[v] != std::numeric_limits<weight_t>::max()) {
        auto source = G.get_source_vertex(min_neighbors[v]);
        auto dest = G.get_destination_vertex(min_neighbors[v]);
        auto weight = min_weights[v];

        // TODO: technically there is a race between reads/writes to
        // roots[dest];
        if (source < dest ||
            G.get_destination_vertex(min_neighbors[roots[dest]]) != source ||
            G.get_source_vertex(min_neighbors[roots[dest]]) != dest) {
          printf("v %i\n", source);
          printf("u %i\n", dest);
          // printf("add mst v %i\n", v);
          // printf("add mst u %i\n", u);
          // printf("add mst v root %i\n", roots[v]);
          // printf("add mst u root %i\n", roots[u]);

          // Not sure cycle comparison for inc/dec vs add; using atomic::add for
          // now because it is in our math.hxx
          math::atomic::add(&mst_weight[0], weight);
          printf("adding source %i dest %i weight %f\n", source, dest, weight);
          printf("mst weight %f\n", mst_weight[0]);
          math::atomic::add(&mst_edges[0], 1);
          math::atomic::add(&super_vertices[0], -1);
          printf("super vertices %i\n", super_vertices[0]);
          atomicExch((&new_roots[v]), new_roots[dest]);
          return;
        }
      }
    };

    // Jump pointers in parallel for
    // TODO: technically there will be races between reads/writes to roots
    // entries, but this will just impact the number of hops
    // read and write from different copies to resolve
    auto jump_pointers_parallel =
        [roots, new_roots] __host__ __device__(vertex_t const& v) -> void {
      vertex_t u = roots[v];
      while (roots[u] != u) {
        u = roots[u];
      }
      new_roots[v] = u;
      return;
    };

    // Execute advance operator to get min weights
    auto in_frontier = &(this->frontiers[0]);
    auto out_frontier = &(this->frontiers[1]);
    operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
        G, get_min_weights, context);

    frontier_t it = *out_frontier;
    // Execute filter operator to get min neighbors
    operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
        G, get_min_neighbors, context);

    // Execute parallel for to add weights to MST
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,           // graph
        add_to_mst,  // lambda function
        context      // context
    );

    // TODO: remove cycles (because we can't check that roots aren't equal when
    // adding due to races)

    // TODO: exit on error if super_vertices not decremented

    // Execute parallel for to jump pointers
    thrust::copy_n(policy, new_roots, P->n_vertices, roots);
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,                       // graph
        jump_pointers_parallel,  // lambda function
        context                  // context
    );
    thrust::copy_n(policy, new_roots, P->n_vertices, roots);
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    if (this->iteration > 1) {
      return true;
    }
    return false;
    // auto P = this->get_problem();
    // return (P->super_vertices[0] == 1);
  }
};

template <typename graph_t>
float run(
    graph_t& G,
    typename graph_t::weight_type* mst_weight,  // Output
    // Context for application (eg, GPU + CUDA stream it will be
    // executed on)
    std::shared_ptr<cuda::multi_context_t> context =
        std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  // instantiate `param` and `result` templates
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  // initialize `param` and `result` w/ the appropriate parameters / data
  // structures
  param_type param;
  result_type result(mst_weight);

  // <boilerplate> This code probably should be the same across all
  // applications, unless maybe you're doing something like multi-gpu /
  // concurrent function calls

  // instantiate `problem` and `enactor` templates.
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  // initialize problem; call `init` and `reset` to prepare data structures
  problem_type problem(G, param, result, context);
  problem.init();
  // problem.reset();

  // initialize enactor; call enactor, returning GPU elapsed time
  enactor_type enactor(&problem, context);
  return enactor.enact();
  // </boilerplate>
}

}  // namespace mst
}  // namespace gunrock