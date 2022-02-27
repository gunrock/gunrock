#pragma once

#include <gunrock/algorithms/algorithms.hxx>

using vertex_t = int;
using edge_t = int;
using weight_t = float;

namespace gunrock {
namespace mst {

// there might be a race here but it would just change the # of jumps
// Does this need context?
void jump_pointers __host__ __device__(vertex_t* roots, vertex_t v) {
  vertex_t u = roots[v];
  while (roots[u] != u) {
    u = roots[u];
  }
  roots[v] = u;
  return;
}

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
  thrust::device_vector<weight_t> min_weights;
  thrust::device_vector<int> mst_edges;
  thrust::device_vector<int> super_vertices;

  void init() {
    auto policy = this->context->get_context(0)->execution_policy();

    roots.resize(n_vertices);
    min_weights.resize(n_vertices);

    mst_edges.resize(1);
    super_vertices.resize(1);

    auto d_mst_weight = thrust::device_pointer_cast(this->result.mst_weight);
    thrust::fill(policy, min_weights.begin(), min_weights.end(),
                 std::numeric_limits<weight_t>::max());
    thrust::fill(policy, d_mst_weight, d_mst_weight + 1, 0);
    thrust::sequence(policy, roots.begin(), roots.end(), 0);
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
    auto roots = P->roots.data().get();
    auto min_weights = P->min_weights.data().get();

    // Get parameters and datastructures

    // Find minimum weight for each vertex
    // TODO: update for multi-directional edges?
    // TODO: fix issue! will return all previous min edges
    auto get_min_weights = [min_weights, roots] __host__ __device__(
                               vertex_t const& source,    // source of edge
                               vertex_t const& neighbor,  // destination of edge
                               edge_t const& edge,        // id of edge
                               weight_t const& weight     // weight of edge
                               ) -> bool {
      // If they are already part of same super vertex, do not check
      if (roots[source] == roots[neighbor]) {
        return false;
      }

      // If they are already part of same super vertex, do not check
      // Store minimum weight
      auto old_dist = math::atomic::min(&(min_weights[source]), weight);

      // If the new distance is better than the previously known
      // best_distance, add `neighbor` to the frontier
      return weight < old_dist;
    };

    // Iterate over edges in new frontier
    // Add weights to MST
    // Update roots
    auto add_to_mst =
        [G, roots, mst_weight, mst_edges, super_vertices] __host__ __device__(
            edge_t const& e) -> void {
      auto v = G.get_source_vertex(e);
      auto u = G.get_destination_vertex(e);
      auto weight = G.get_edge_weight(e);

      // TODO: need total ordering but this is restricting some edges that
      // should be added; need to check for dup and not exclude if there is no
      // dup; checking for dup edge in frontier should work
      printf("v %i\n", v);
      printf("u %i\n", u);
      if (v < u) {
        // printf("add mst v %i\n", v);
        // printf("add mst u %i\n", u);
        // printf("add mst v root %i\n", roots[v]);
        // printf("add mst u root %i\n", roots[u]);

        jump_pointers(roots, v);
        jump_pointers(roots, u);

        // Not sure cycle comparison for inc/dec vs add; using atomic::add for
        // now because it is in our math.hxx
        math::atomic::add(&mst_weight[0], weight);
        printf("weight %f\n", mst_weight[0]);
        math::atomic::add(&mst_edges[0], 1);
        math::atomic::add(&super_vertices[0], -1);
        atomicExch((&roots[roots[v]]), roots[u]);
        return;
      }
    };

    // Jump pointers in parallel for
    // I do not think parallel updates to roots will cause issues but need to
    // think about more; there might be a race here but it would just change the
    // # of jumps
    auto jump_pointers_parallel =
        [roots] __host__ __device__(vertex_t const& v) -> void {
      vertex_t u = roots[v];
      while (roots[u] != u) {
        u = roots[u];
      }
      roots[v] = u;
      return;
    };

    // Execute advance operator to get min weights
    auto in_frontier = &(this->frontiers[0]);
    auto out_frontier = &(this->frontiers[1]);
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::edges,
                                operators::advance_io_type_t::edges>(
        G, get_min_weights, in_frontier, out_frontier, E->scanned_work_domain,
        context);

    // Execute parallel for to add weights to MST
    // TODO: ensure this executes on new frontier outputted from advance above
    operators::parallel_for::execute<operators::parallel_for_each_t::element>(
        *out_frontier,  // graph
        add_to_mst,     // lambda function
        context         // context
    );

    // TODO: remove duplicates (increment super vertices when removing)

    // TODO: exit on error if super_vertices not decremented

    // Execute parallel for to jump pointers
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,                       // graph
        jump_pointers_parallel,  // lambda function
        context                  // context
    );
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    // TODO: update condition
    if (this->iteration > 0) {
      return true;
    }
    return false;
    // auto P = this->get_problem();
    // return *(P->super_vertices) == 1;
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