#pragma once
#include <gunrock/container/experimental/async/queue.hxx>

namespace gunrock {
namespace experimental {
namespace async {
template <typename algorithm_problem_t>
struct enactor_t {
  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t = typename algorithm_problem_t::edge_t;
  using weight_t = typename algorithm_problem_t::weight_t;

  using single_queue_t = uint32_t;
  using queue_t = Queues<vertex_t, single_queue_t>;
  queue_t q;

  algorithm_problem_t* problem;
  std::shared_ptr<gcuda::multi_context_t> context;

  enactor_t(const enactor_t& rhs) = delete;
  enactor_t& operator=(const enactor_t& rhs) = delete;

  // !! These parameters may be application specific, so really we want a way to
  // pass them in as "execution policy parameters"
  // For instance ... I don't think these parameters would work w/ async
  // pagerank
  int num_queue = 4;
  int min_iter = 800;
  int num_block = 56 * 5;
  int num_thread = 256;
  float sizing_factor = 1.5;

  enactor_t(algorithm_problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : problem(_problem), context(_context) {
    auto n_vertices = problem->get_graph().get_number_of_vertices();

    auto capacity = min(
        single_queue_t(1 << 30),
        max(single_queue_t(1024), single_queue_t(n_vertices * sizing_factor)));

    q.init(capacity, num_queue, num_block, num_thread, min_iter);
    q.reset();
  }

  enactor_t* get_enactor() { return this; }
  algorithm_problem_t* get_problem() { return problem; }

  virtual void loop(gcuda::multi_context_t& context) = 0;
  virtual void prepare_frontier(queue_t& q,
                                gcuda::multi_context_t& context) = 0;

  float enact() {
    auto single_context = context->get_context(0);
    prepare_frontier(q, *context);
    auto timer = single_context->timer();
    timer.begin();
    loop(*context);
    q.sync();  // !! Best way to synchronize w/ the `context` stream?  Do we
               // need to?
    return timer.end();
  }
};

}  // namespace async
}  // namespace experimental
}  // namespace gunrock