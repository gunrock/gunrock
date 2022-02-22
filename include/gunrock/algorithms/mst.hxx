#pragma once

#include <gunrock/algorithms/algorithms.hxx>

using vertex_t = int;
using edge_t = int;
using weight_t = float;

namespace gunrock {
namespace mst {

struct super {
  super* root;
  float min_weight;
  vertex_t min_neighbor;
};

// there might be a race here but it would just change the # of jumps
// Does this need context?
void jump_pointers __host__ __device__(super* supers, vertex_t v) {
  super* u = supers[v].root;
  while (u->root != u) {
    u = u->root;
  }
  supers[v].root = u;
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

  int* mst_edges;
  int* super_vertices;
  struct super* supers = new super[n_vertices];

  void init() {
    *(result.mst_weight) = 0;
    *mst_edges = 0;
    *super_vertices = n_vertices;
  }

  void reset() {
    *(result.mst_weight) = 0;
    *mst_edges = 0;
    *super_vertices = n_vertices;
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

    // Fill the frontier with a sequence of vertices from 0 -> n_vertices.
    f->sequence((vertex_t)0, P->n_vertices, context.get_context(0)->stream());
  }

  // One iteration of the application
  void loop(cuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto supers = P->supers;
    auto mst_weight = P->result.mst_weight;
    auto mst_edges = P->mst_edges;
    auto super_vertices = P->super_vertices;

    // Get parameters and datastructures
    int old_super_vertices = *super_vertices;

    // TODO: switch DSs to thrust vectors, move initialization to init/reset and
    // use thrust::fill and thrust::for_each
    if (this->iteration == 0) {
      // Parallel for to set root and min_weight (in supers) for each vertex
      auto set_supers =
          [supers] __host__ __device__(vertex_t const& v) -> void {
        supers[v].root = supers + v;
        supers[v].min_weight = std::numeric_limits<weight_t>::max();
      };
      operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
          G,           // graph
          set_supers,  // lambda function
          context      // context
      );

    } else {
      // Parallel for to set min_weight (in supers) for each vertex
      auto set_supers =
          [supers] __host__ __device__(vertex_t const& v) -> void {
        supers[v].min_weight = std::numeric_limits<weight_t>::max();
      };
      operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
          G,           // graph
          set_supers,  // lambda function
          context      // context
      );
    }

    // Find minimum weight for each vertex 
    // TODO: Update for multi-directional edges?
    auto min_weights = [supers] __host__ __device__(
                           vertex_t const& source,    // source of edge
                           vertex_t const& neighbor,  // destination of edge
                           edge_t const& edge,        // id of edge
                           weight_t const& weight     // weight of edge
                           ) -> bool {
      // If they are already part of same super vertex, do not check
      if (supers[source].root == supers[neighbor].root) {
        return false;
      }
      // Store minimum weight
      weight_t old_dist =
          math::atomic::min(&(supers[source].min_weight), weight);

      // If the new distance is better than the previously known
      // best_distance, add `neighbor` to the frontier
      return weight < old_dist;
    };

    // Iterate over edges in new frontier
    // Add weights to MST
    // Update roots
    auto add_to_mst =
        [G, context, supers, mst_weight, mst_edges, super_vertices] __host__
        __device__(edge_t const& e) -> void {
      auto v = G.get_source_vertex(e);
      auto u = G.get_destination_vertex(e);
      auto weight = G.get_edge_weight(e);

      // Jump pointers for source and destination to get accurate roots
      jump_pointers(supers, v);
      jump_pointers(supers, u);

      // Do I want to use all math::atomic::add to fit into math or should I use
      // inc/dec atomic cuda functions or should I add those to math.hxx? Not
      // sure cycle comparison for inc/dec vs add
      math::atomic::add(mst_weight, weight);
      math::atomic::add(mst_edges, 1);
      math::atomic::add(super_vertices, -1);

      supers[v].root->root = supers[u].root;
      // Is atomic exch needed to resolve race between updates to v root and u?
      // I don't think so; if it gets wrong root, will be resolved during
      // pointer jumping
      // cuda::atomicExch(&(supers[v].root->root),supers[u].root);
      return;
    };

    // Jump pointers in parallel for
    // I do not think parallel updates to roots will cause issues but need to
    // think about more there might be a race here but it would just change the
    // # of jumps
    auto jump_pointers_parallel =
        [supers] __host__ __device__(vertex_t const& v) -> void {
      super* u = supers[v].root;
      while (u->root != u) {
        u = u->root;
      }
      supers[v].root = u;
      return;
    };

    // Execute advance operator to get min weights
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::edges,
                                operators::advance_io_type_t::edges>(
        G, E, min_weights, context);

    // Execute parallel for to add weights to MST
    operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
        G,           // graph
        add_to_mst,  // lambda function
        context      // context
    );

    // TODO: remove duplicates (increment super vertices when removing)

    // Execute parallel for to jump pointers 
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,                       // graph
        jump_pointers_parallel,  // lambda function
        context                  // context
    );

    // Exit on error if super_vertices not decremented
    if (*super_vertices == old_super_vertices) {
      std::cout << "error\n";
    }
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    auto P = this->get_problem();
    return *(P->super_vertices) == 1;
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
  problem.reset();

  // initialize enactor; call enactor, returning GPU elapsed time
  enactor_type enactor(&problem, context);
  return enactor.enact();
  // </boilerplate>
}

}  // namespace mst
}  // namespace gunrock