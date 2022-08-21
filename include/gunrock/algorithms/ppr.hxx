/**
 * @file ppr.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Personalized Page-Rank.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace ppr {

template <typename vertex_t, typename weight_t>
struct param_t {
  vertex_t seed;
  weight_t alpha;
  weight_t epsilon;
  param_t(vertex_t _seed, weight_t _alpha, weight_t _epsilon)
      : seed(_seed), alpha(_alpha), epsilon(_epsilon) {}
};

template <typename weight_t>
struct result_t {
  weight_t* p;
  result_t(weight_t* _p) : p(_p) {}
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

  thrust::device_vector<weight_t> r;
  thrust::device_vector<weight_t> r_prime;

  weight_t _2a1a;
  weight_t _1a1a;

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    r.resize(n_vertices);
    r_prime.resize(n_vertices);

    auto alpha = this->param.alpha;
    _2a1a = (2 * alpha) / (1 + alpha);
    _1a1a = ((1 - alpha) / (1 + alpha));
  }

  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    auto context = this->get_single_context();
    auto policy = context->execution_policy();

    auto seed = this->param.seed;
    auto d_p = thrust::device_pointer_cast(this->result.p);

    // Reset everything to 0.
    thrust::fill(policy, d_p, d_p + n_vertices, 0);
    thrust::fill(policy, r.begin(), r.end(), 0);
    thrust::fill(policy, r_prime.begin(), r_prime.end(), 0);

    // Set the seed active.
    thrust::fill(policy, r.begin() + seed, r.begin() + seed + 1, 1);
    thrust::fill(policy, r_prime.begin() + seed, r_prime.begin() + seed + 1, 1);
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
    f->push_back(P->param.seed);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    weight_t* p = P->result.p;
    weight_t* r = P->r.data().get();
    weight_t* r_prime = P->r_prime.data().get();

    auto n_vertices = G.get_number_of_vertices();
    weight_t epsilon = P->param.epsilon;
    weight_t _2a1a = P->_2a1a;
    weight_t _1a1a = P->_1a1a;

    auto filter_op = [p, r, r_prime, _2a1a] __host__ __device__(
                         vertex_t const& vertex) -> bool {
      p[vertex] += _2a1a * r[vertex];
      r_prime[vertex] = 0;
      return true;
    };

    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, filter_op, context);

    auto advance_op = [G, r, r_prime, _1a1a, epsilon] __host__ __device__(
                          vertex_t const& src, vertex_t const& dst,
                          edge_t const& edge, weight_t const& weight) -> bool {
      auto update = _1a1a * r[src] / (weight_t)G.get_number_of_neighbors(src);
      auto oldval = math::atomic::add(r_prime + dst, update);
      auto newval = oldval + update;
      auto thresh = (weight_t)G.get_number_of_neighbors(dst) * epsilon;
      return (oldval < thresh) && (newval >= thresh);
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, advance_op, context);

    auto policy = this->context->get_context(0)->execution_policy();
    thrust::copy_n(policy, r_prime, n_vertices, r);
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& seed,
          typename graph_t::weight_type* p,
          typename graph_t::weight_type& alpha,
          typename graph_t::weight_type& epsilon,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t, weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(seed, alpha, epsilon);
  result_type result(p);
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

template <typename graph_t>
float run_batch(graph_t& G,
                typename graph_t::vertex_type& n_seeds,
                typename graph_t::weight_type* p,
                typename graph_t::weight_type& alpha,
                typename graph_t::weight_type& epsilon) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  auto n_vertices = G.get_number_of_vertices();
  auto f = [&](std::size_t job) -> float {
    vertex_t active_seed = job;
    weight_t* _p = p + (n_vertices * active_seed);
    return ppr::run(G, active_seed, _p, alpha, epsilon);
  };

  thrust::host_vector<float> total_elapsed(1);
  operators::batch::execute(f, n_seeds, total_elapsed.data());

  return total_elapsed[0];
}

}  // namespace ppr
}  // namespace gunrock