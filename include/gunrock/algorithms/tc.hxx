/**
 * @file sssp.hxx
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

namespace gunrock {
namespace tc {

template <typename vertex_t>
struct param_t {
  // No parameters for this algorithm
};

template <typename vertex_t>
struct result_t {
  vertex_t* triangles_count;
  result_t(vertex_t* _triangles_count) : triangles_count(_triangles_count) {}
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

    f->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto triangles_count = P->result.triangles_count;
    auto iteration = this->iteration;

    auto intersect = [G, triangles_count] __host__ __device__(
                         vertex_t const& source,    // ... source
                         vertex_t const& neighbor,  // neighbor
                         edge_t const& edge,        // edge
                         weight_t const& weight     // weight (tuple).
                         ) -> bool {
      if (source < neighbor) {
        auto src_triangles_count = G.get_intersection_count(source, neighbor);
        math::atomic::add(&(triangles_count[source]), src_triangles_count);
      }
      return false;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, intersect, context);
    std::cout << "iteration: " << iteration << std::endl;
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type* triangles_count,  // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param;
  result_type result(triangles_count);
  // </user-defined>

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace tc
}  // namespace gunrock