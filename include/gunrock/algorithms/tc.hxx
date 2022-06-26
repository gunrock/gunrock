/**
 * @file tc.hxx
 * @author Muhammad A. Awad (mawad@ucdavis.edu)
 * @brief Triangle Counting algorithm.
 * @version 0.1
 * @date 2022-08-06
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/util/timer.hxx>

namespace gunrock {
namespace tc {

template <typename vertex_t>
struct param_t {
  bool reduce_all_triangles;
  param_t(bool _reduce_all_triangles)
      : reduce_all_triangles(_reduce_all_triangles) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* vertex_triangles_count;
  std::size_t* total_triangles_count;
  result_t(vertex_t* _vertex_triangles_count, uint64_t* _total_triangles_count)
      : vertex_triangles_count(_vertex_triangles_count),
        total_triangles_count(_total_triangles_count) {}
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

  void init() override {}

  void reset() override {}
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties)
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto vertex_triangles_count = P->result.vertex_triangles_count;

    auto intersect = [G, vertex_triangles_count] __host__ __device__(
                         vertex_t const& source,    // ... source
                         vertex_t const& neighbor,  // neighbor
                         edge_t const& edge,        // edge
                         weight_t const& weight     // weight (tuple).
                         ) -> bool {
      if (neighbor > source) {
        auto src_vertex_triangles_count = G.get_intersection_count(
            source, neighbor,
            [vertex_triangles_count, source,
             neighbor](auto intersection_vertex) {
              if (source != intersection_vertex &&
                  neighbor != intersection_vertex) {
                math::atomic::add(
                    &(vertex_triangles_count[intersection_vertex]),
                    vertex_t{1});
              }
            });
      }
      return false;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        G, E, intersect, context);
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    if (this->iteration == 1)
      return true;
    return false;
  }

  float post_process() {
    util::timer_t timer;
    timer.begin();
    auto P = this->get_problem();
    auto G = P->get_graph();

    if (P->param.reduce_all_triangles) {
      auto policy = this->context->get_context(0)->execution_policy();
      *P->result.total_triangles_count = thrust::transform_reduce(
          policy, P->result.vertex_triangles_count,
          P->result.vertex_triangles_count + G.get_number_of_vertices(),
          [] __device__(const vertex_t& vertex_triangles) {
            return static_cast<std::size_t>(vertex_triangles);
          },
          std::size_t{0}, thrust::plus<std::size_t>());
    }
    return timer.end();
  }
};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          bool reduce_all_triangles,
          typename graph_t::vertex_type* vertex_triangles_count,  // Output
          std::size_t* total_triangles_count,                     // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param(reduce_all_triangles);
  result_type result(vertex_triangles_count, total_triangles_count);
  // </user-defined>

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;
  enactor_type enactor(&problem, context, props);
  auto time = enactor.enact();
  time += enactor.post_process();

  // </boiler-plate>
  return time;
}

}  // namespace tc
}  // namespace gunrock