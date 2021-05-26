#pragma once

#include <bits/stdc++.h>

#include <gunrock/algorithms/algorithms.hxx>
#include "gunrock/algorithms/async/queue.cuh"

namespace gunrock {
namespace async {
namespace bfs {

// <user-defined>
// OK
template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};
// </user-defined>

// <user-defined>
// OK
template <typename edge_t>
struct result_t {
  edge_t* depth;
  result_t(edge_t* _depth) : depth(_depth) {}
};
// </user-defined>

// This is very close to compatible w/ standard Gunrock problem_t
// However, it doesn't use the `context` argument, so not joining yet
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
  
  void init() override {}
  
  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    
    auto context = this->get_single_context();
    auto policy  = context->execution_policy();
    
    auto single_source = param.single_source;
    auto d_depth       = thrust::device_pointer_cast(this->result.depth);
    thrust::fill(policy, d_depth + 0, d_depth + n_vertices, n_vertices + 1);
    thrust::fill(policy, d_depth + single_source, d_depth + single_source + 1, 0);
  }
};

// --
// Enactor

// !! This is annoying ... ideally we'd be able to initialize the frontier
//    w/ a lambda or thrust call.  But not sure how to get that to work.
template<typename queue_t, typename val_t>
__global__ void _push_one(queue_t q, val_t val) {
    q.push(val);
}

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t   = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  
  using single_queue_t = uint32_t;
  using queue_t        = MaxCountQueue::Queues<vertex_t, single_queue_t>;
  queue_t q;
  
  int num_queue  = 4;
  int min_iter   = 800;
  int num_block  = 56 * 5;
  int num_thread = 256;
  
  // !! Have to implement stubs for these
  void prepare_frontier(frontier_t<vertex_t>* f, cuda::multi_context_t& context) override {}
  void loop(cuda::multi_context_t& context) override {}
  
  void prepare_frontier(queue_t& q, cuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto n_vertices = P->get_graph().get_number_of_vertices();
   
    // Initialize queues
    // !! Maybe this should go somewhere else?
    auto capacity = min(
      single_queue_t(1 << 30), 
      max(single_queue_t(1024), single_queue_t(n_vertices * 1.5))
    );
    
    q.init(capacity, num_queue, min_iter);
    q.reset();
    
    // !! MaxCountQueue::Queues creates it's own streams.  But I think we should at least
    //    synchronizing the to the `context` stream?
    _push_one<<<1, 1>>>(q, P->param.single_source);
  }
  
  float enact() {
    auto context = this->context;
    auto single_context = context->get_context(0);
    prepare_frontier(q, *context);
    auto timer = single_context->timer();
    timer.begin();
    async_loop(*context);
    // finalize(*context);
    return timer.end();
  }
  
  void async_loop(cuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto G = P->get_graph();
    
    edge_t* depth = P->result.depth;
    
    auto async_bfs_op = [G, depth] __device__ (vertex_t node, queue_t q) -> void {
        
        vertex_t d = ((volatile vertex_t * )depth)[node];
        
        const vertex_t start  = G.get_starting_edge(node);
        const vertex_t degree = G.get_number_of_neighbors(node);
        
        for(int idx = 0; idx < degree; idx++) {
            vertex_t neib  = G.get_destination_vertex(start + idx);
            vertex_t old_d = atomicMin(depth + neib, d + 1);
            if(old_d > d + 1)
                q.push(neib);
        }
    };
    
    q.launch_thread(num_block, num_thread, async_bfs_op);
    q.sync();
    // !! Best way to synchronize w/ the `context` stream?  Do we need to?
  }
};


template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::edge_type* depth             // Output
) {
  
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;

  using param_type   = param_t<vertex_t>;
  using result_type  = result_t<edge_t>;
  
  param_type param(single_source);
  result_type result(depth);
  // </user-defined>
  
  // <boiler-plate>
  auto multi_context =
      std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, multi_context);
  problem.init();
  problem.reset();
  
  enactor_type enactor(&problem, multi_context);
  return enactor.enact();
  // </boiler-plate>
}

} // namespace bfs
} // namespace async
} // namespace gunrock