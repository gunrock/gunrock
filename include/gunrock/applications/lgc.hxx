/**
 * @file lgc.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Original implementation by: Ben Johnson (bkj.322@gmail.com)
 * @version 0.1
 * @date 2021-05-03
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/applications/application.hxx>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace gunrock {
namespace lgc {

template <typename vertex_t, typename weight_t>
struct param_t {
  graph::vertex_pair_t<vertex_t> pair;

  weight_t eps;            // Tolerance for convergence
  weight_t alpha;          // Parameterizes conductance/size of output cluster
  weight_t rho;            // Parameterizes conductance/size of output cluster
  int maximum_iterations;  // Maximum number of iterations

  param_t(vertex_t source,
          vertex_t source_neighbor,
          weight_t _eps,
          weight_t _alpha,
          weight_t _rho)
      : pair.source(_source),
      pair.destination(_source_neighbor), eps(_eps), alpha(_alpha), rho(_rho) {}
};

template <typename weight_t>
struct result_t {
  weight_t* q;  // Truncated z-values && also the output.
  result_t(weight_t* _q) : q(_q) {}
};

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

  thrust::device_vector<weight_t> gradient;  // Gradient values
  thrust::device_vector<weight_t> y;         // Intermediate quantity
  thrust::device_vector<weight_t> z;         // Intermediate quantity
  thrust::device_vector<int> visited;        // track

  thrust::device_vector<int> grad_scale(1, 0);
  thrust::device_vector<weight_t> grad_scale_value(1, (weight_t)0);

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    gradient.resize(n_vertices);
    y.resize(n_vertices);
    z.resize(n_vertices);
    q.resize(n_vertices);
    visited.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    thrust::fill_n(policy, gradient.begin(), n_vertices, weight_t(0));
    thrust::fill_n(policy, y.begin(), n_vertices, weight_t(0));
    thrust::fill_n(policy, z.begin(), n_vertices, weight_t(0));
    thrust::fill_n(policy, visited.begin(), n_vertices, 0);

    thrust::fill_n(policy, this->result.q, n_vertices, weight_t(0));
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void prepare_frontier(frontier_t<vertex_t>* f,
                        cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.pair.source);
    f->push_back(P->param.pair.destination);
  }

  void loop(cuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto f = E->get_input_frontier();

    auto n_vertices = G.get_number_of_vertices();
    auto q = P->result.q;

    auto gradient = P->gradient.data().get();
    auto y = P->y.data().get();
    auto z = P->z.data().get();
    auto visited = P->visited.data().get();

    auto alpha = P->param.alpha;
    auto rho = P->param.rho;

    auto pair = P->param.pair;

    int num_ref_nodes = 1;

    auto policy = this->context->get_context(0)->execution_policy();

    // compute operation
    auto compute_op = [=] __host__ __device__(vertex_t const& v) {
      // ignore the neighbor on the first iteration
      if ((iteration == 0) && (v == pair.destination))
        return;

      // Compute degrees
      auto degree = G.get_number_of_neighbors(v);
      auto degree_sqrt = sqrt((weight_t)degree);
      auto inv_degree_sqrt = 1.0 / degree_sqrt;

      // this is at end in original implementation, but works
      // here after the first iteration (+ have to adjust for
      // it in StopCondition)
      if ((iteration > 0) && (v == pair.source)) {
        gradient[v] -= alpha / num_ref_nodes * inv_degree_sqrt;
      }

      z[v] = y[v] - gradient[v];

      if (z[v] == 0)
        return;

      auto q_old = q[v];
      auto thresh = rho * alpha * degree_sqrt;

      if (z[v] >= thresh) {
        q[v] = z[v] - thresh;
      } else if (z[v] <= -thresh) {
        q[v] = z[v] + thresh;
      } else {
        q[v] = (weight_t)0;
      }

      if (iteration == 0) {
        y[v] = q[v];
      } else {
        auto beta = (1 - sqrt(alpha)) / (1 + sqrt(alpha));
        y[v] = q[v] + beta * (q[v] - q_old);
      }

      visited[v] = false;
      gradient[v] = y[v] * (1.0 + alpha) / 2;

      return 0;  // ignored.
    };

    thrust::transform(policy, f->begin(), f->end(),
                      thrust::make_discard_iterator(), compute_op);

    auto spread_op = [=] __host__ __device__(
                         vertex_t const& src, vertex_t const& dst,
                         edge_t const& edge, weight_t const& weight) -> bool {
      weight_t src_dn_sqrt =
          1.0 / sqrt((weight_t)G.get_number_of_neighbors(src));
      weight_t dest_dn_sqrt =
          1.0 / sqrt((weight_t)G.get_number_of_neighbors(dst));
      weight_t src_y = y[src];

      weight_t grad_update =
          -src_dn_sqrt * src_y * dest_dn_sqrt * (1.0 - alpha) / 2;
      weight_t last_grad = math::atomic::add(gradient + dst, grad_update);
      if (last_grad + grad_update == 0)
        return false;

      bool already_touched = math::atomic::max(visited + dst, 1) == 1;
      return !already_touched;
    };

    operators::advance::execute<operators::load_balance_t::merge_path>(
        G, E, spread_op, context);
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    // never break on first iteration
    if (this->iteration == 0)
      return false;

    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto f = E->get_input_frontier();

    auto n_vertices = G.get_number_of_vertices();

    auto gradient = P->gradient.data().get();

    auto alpha = P->param.alpha;
    auto rho = P->param.rho;
    auto eps = P->param.eps;
    auto pair = P->param.pair;

    auto q = P->result.q;

    auto d_grad_scale = P->grad_scale.data();
    auto d_grad_scale_value = P->grad_scale_value.data();

    weight_t grad_thresh = rho * alpha * (1 + eps);

    auto convergence_op = [=] __host__ __device__(vertex_t const& v) {
      weight_t v_dn_sqrt = 1.0 / sqrt((weight_t)G.get_number_of_neighbors(v));
      weight_t val = gradient[v];

      if (v == pair.source)
        val -= (alpha / num_ref_nodes) * v_dn_sqrt;

      val = abs(val * v_dn_sqrt);

      math::atomic::max(d_grad_scale_value, val);
      if (val > grad_thresh) {
        math::atomic::max(d_grad_scale, 1);
      }

      return 0;  // ignored.
    };

    auto policy = this->context->get_context(0)->execution_policy();
    thrust::transform(policy, f->begin(), f->end(),
                      thrust::make_discard_iterator(), convergence_op);

    thrust::host_vector<int> check_grad_scale = grad_scale;
    thrust::host_vector<weight_t> check_grad_scale_value = grad_scale_value;

    // gradient too small:: converged.
    if (!(check_grad_scale[0])) {
      auto n_vertices = G.get_number_of_vertices();
      auto scale_op = [=] __device__ __host__(const vertex_t& v) {
        return abs(q[v] * sqrt((weight_t)G.get_number_of_neighbors(v)));
      };

      thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                        thrust::counting_iterator<vertex_t>(n_vertices), q,
                        scale_op);
      return true;
    }

    return false;
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type source,
          typename graph_t::vertex_type source_neighbor,
          typename graph_t::weight_type eps,
          typename graph_t::weight_type alpha,
          typename graph_t::weight_type rho,
          typename graph_t::weight_type* q  // Output
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t, weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(source, source_neighbor, eps, alpha, rho);
  result_type result(q);
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

}  // namespace lgc
}  // namespace gunrock