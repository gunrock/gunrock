#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/framework/experimental/async/enactor.hxx>

namespace gunrock {
namespace experimental {
namespace async {
namespace bfs {

// !! Identical to synchronous `essentials`
template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

// !! Identical to synchronous `essentials`
template <typename edge_t>
struct result_t {
  edge_t* depth;
  result_t(edge_t* _depth) : depth(_depth) {}
};

// !! Identical to synchronous `essentials`
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  void init() override {
    // noop
  }

  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    auto context = this->get_single_context();
    auto policy = context->execution_policy();

    auto single_source = param.single_source;
    auto d_depth = thrust::device_pointer_cast(this->result.depth);
    thrust::fill_n(policy, d_depth, n_vertices,
                   std::numeric_limits<vertex_t>::max());
    thrust::fill_n(policy, d_depth + single_source, 1, 0);
  }
};

// --
// Enactor

// !! This is annoying ... ideally we'd be able to initialize the frontier
//    w/ a lambda or thrust call.  But not sure how to get that to work.
template <typename queue_t, typename val_t>
__global__ void _push_one(queue_t q, val_t val) {
  q.push(val);
}

template <typename problem_t>
struct enactor_t : async::enactor_t<problem_t> {
  using async::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using queue_t = typename async::enactor_t<problem_t>::queue_t;

  // !! Breaks w/ standard essentials (mildly...)
  void prepare_frontier(queue_t& q, gcuda::multi_context_t& context) {
    auto P = this->get_problem();

    // !! Queues creates it's own streams.  But I think we should at least
    //    synchronizing the to the `context` stream?
    _push_one<<<1, 1>>>(q, P->param.single_source);
  }

  // !! Breaks w/ standard essentials (mildly...)
  void loop(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto q = this->q;

    edge_t* depth = P->result.depth;

    q.launch_thread([G, depth] __device__(vertex_t node, queue_t q) -> void {
      vertex_t d = ((volatile vertex_t*)depth)[node];

      const vertex_t start = G.get_starting_edge(node);
      const vertex_t degree = G.get_number_of_neighbors(node);

      for (int idx = 0; idx < degree; idx++) {
        vertex_t neib = G.get_destination_vertex(start + idx);
        vertex_t old_d = atomicMin(depth + neib, d + 1);
        if (old_d > d + 1)
          q.push(neib);
      }
    });
  }
};

// !! Identical to synchronous `essentials`
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::edge_type* depth             // Output
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<edge_t>;

  param_type param(single_source);
  result_type result(depth);
  // </user-defined>

  // <boiler-plate>
  auto multi_context =
      std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0));

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, multi_context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, multi_context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace bfs
}  // namespace async
}  // namespace experimental
}  // namespace gunrock