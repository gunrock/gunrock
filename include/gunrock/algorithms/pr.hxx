/**
 * @file pr.hxx
 * @author Ben Johnson (bkj.322@gmail.com)
 * @brief PageRank
 * @date 2021-04-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>

namespace gunrock {
namespace pr {

template <typename weight_t>
struct param_t {
  weight_t alpha;
  weight_t tol;
  bool collect_metrics;

  param_t(weight_t _alpha, weight_t _tol, bool _collect_metrics)
      : alpha(_alpha), tol(_tol), collect_metrics(_collect_metrics) {}
};

template <typename weight_t>
struct result_t {
  weight_t* p;
  int* search_depth;
  result_t(weight_t* _p, int* _search_depth)
      : p(_p), search_depth(_search_depth) {}
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

  thrust::device_vector<weight_t>
      plast;  // pagerank values from previous iteration
  thrust::device_vector<weight_t>
      iweights;  // alpha * 1 / (sum of outgoing weights) -- used to determine
                 // out of mass spread from src to dst

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    plast.resize(n_vertices);
    iweights.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();

    auto g = this->get_graph();

    auto n_vertices = g.get_number_of_vertices();
    auto alpha = this->param.alpha;

    thrust::fill_n(policy, this->result.p, n_vertices, 1.0 / n_vertices);

    thrust::fill_n(policy, plast.begin(), n_vertices, 0);

    auto get_weight = [=] __device__(const int& i) -> weight_t {
      weight_t val = 0;

      edge_t start = g.get_starting_edge(i);
      edge_t end = start + g.get_number_of_neighbors(i);
      for (edge_t offset = start; offset < end; offset++) {
        val += g.get_edge_weight(offset);
      }

      return val != 0 ? alpha / val : 0;
    };

    thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                      thrust::counting_iterator<vertex_t>(n_vertices),
                      iweights.begin(), get_weight);

    *(this->result.search_depth) = 0;
  }
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

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();
    auto p = P->result.p;
    auto plast = P->plast.data().get();
    auto iweights = P->iweights.data().get();
    auto alpha = P->param.alpha;

    auto policy = this->context->get_context(0)->execution_policy();

    auto search_depth = P->result.search_depth;

    auto collect_metrics = P->param.collect_metrics;

    thrust::copy_n(policy, p, n_vertices, plast);

    // >> handle "dangling nodes" (nodes w/ zero outdegree)
    // could skip this if no nodes have sero outdegree
    auto compute_dangling = [=] __device__(const int& i) -> weight_t {
      return iweights[i] == 0 ? alpha * p[i] : 0;
    };

    float dsum = thrust::transform_reduce(
        policy, thrust::counting_iterator<vertex_t>(0),
        thrust::counting_iterator<vertex_t>(n_vertices), compute_dangling,
        (weight_t)0.0, thrust::plus<weight_t>());

    thrust::fill_n(policy, p, n_vertices, (1 - alpha + dsum) / n_vertices);
    // -- OR --
    // skip dangling nodes
    // thrust::fill_n(policy,
    //   p, n_vertices, (1 - alpha) / n_vertices);
    // <<

    auto spread_coo_op = [=] __device__(edge_t const& e) -> void {
      auto src = G.get_source_vertex(e);
      auto dst = G.get_destination_vertex(e);
      weight_t weight = G.get_edge_weight(e);
      weight_t update = plast[src] * iweights[src] * weight;
      math::atomic::add(p + dst, update);
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
        G,              // graph
        spread_coo_op,  // lambda function
        context         // context
    );

    // -- OR --
    // You can do the following with advance operator instead:
    // auto spread_op = [=] __host__ __device__(
    //                      vertex_t const& src, vertex_t const& dst,
    //                      edge_t const& edge, weight_t const& weight) -> bool
    // {
    //   weight_t update = plast[src] * iweights[src] * weight;
    //   math::atomic::add(p + dst, update);
    //   return false;
    // };
    // operators::advance::execute<operators::load_balance_t::block_mapped,
    //                             operators::advance_direction_t::forward,
    //                             operators::advance_io_type_t::graph,
    //                             operators::advance_io_type_t::none>(
    //     G, E, spread_op, context);
    // <<

    if (collect_metrics) {
      *search_depth = this->iteration;
    }
  }

  virtual bool is_converged(gcuda::multi_context_t& context) override {
    if (this->iteration == 0)
      return false;

    auto P = this->get_problem();
    auto G = P->get_graph();
    auto tol = P->param.tol;

    auto n_vertices = G.get_number_of_vertices();
    auto p = P->result.p;
    auto plast = P->plast.data().get();

    auto abs_diff = [=] __device__(const int& i) -> weight_t {
      return abs(p[i] - plast[i]);
    };

    auto policy = this->context->get_context(0)->execution_policy();
    float err = thrust::transform_reduce(
        policy, thrust::counting_iterator<vertex_t>(0),
        thrust::counting_iterator<vertex_t>(n_vertices), abs_diff,
        (weight_t)0.0, thrust::maximum<weight_t>());

    return err < tol;
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type alpha,
          typename graph_t::weight_type tol,
          bool collect_metrics,
          typename graph_t::weight_type* p,  // Output
          int* search_depth,                 // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(alpha, tol, collect_metrics);
  result_type result(p, search_depth);
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
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace pr
}  // namespace gunrock