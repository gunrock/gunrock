
#pragma once

#include <gunrock/algorithms/async/queue.cuh>

namespace gunrock {
  
template <typename algorithm_problem_t>
struct async_enactor_t {

  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t   = typename algorithm_problem_t::edge_t;
  using weight_t = typename algorithm_problem_t::weight_t;
  
  using single_queue_t = uint32_t;
  using queue_t        = MaxCountQueue::Queues<vertex_t, single_queue_t>;
  queue_t q;
  
  algorithm_problem_t* problem;
  std::shared_ptr<cuda::multi_context_t> context;

  async_enactor_t(const async_enactor_t& rhs) = delete;
  async_enactor_t& operator=(const async_enactor_t& rhs) = delete;

  // !! These parameters may be application specific, so really we want a way to 
  // pass them in as "execution policy parameters"
  // For instance ... I don't think these parameters would work w/ async pagerank
  int num_queue       = 4;
  int min_iter        = 800;
  int num_block       = 56 * 5;
  int num_thread      = 256;
  float sizing_factor = 1.5;

  async_enactor_t(algorithm_problem_t* _problem,
            std::shared_ptr<cuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
  : problem(_problem),
    context(_context) {

    auto n_vertices = problem->get_graph().get_number_of_vertices();
    
    auto capacity = min(
      single_queue_t(1 << 30), 
      max(single_queue_t(1024), single_queue_t(n_vertices * sizing_factor))
    );
    
    q.init(capacity, num_queue, num_block, num_thread, min_iter);
    q.reset();
  }
  
  async_enactor_t*     get_enactor() { return this;    }
  algorithm_problem_t* get_problem() { return problem; }
  
  virtual void loop(cuda::multi_context_t& context)                         = 0;
  virtual void prepare_frontier(queue_t& q, cuda::multi_context_t& context) = 0;

  float enact() {
    auto single_context = context->get_context(0);
    prepare_frontier(q, *context);
    auto timer = single_context->timer();
    timer.begin();
    loop(*context);
    q.sync(); // !! Best way to synchronize w/ the `context` stream?  Do we need to?
    return timer.end();
  }
};

}  // namespace gunrock