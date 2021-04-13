/**
 * @file pr.hxx
 * @author Ben Johnson (bkj.322@gmail.com)
 * @brief PageRank
 * @version 0.1
 * @date 2021-04-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/applications/application.hxx>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>

namespace gunrock {
namespace pr {

template <typename weight_t>
struct param_t {
  weight_t alpha;
  weight_t tol;
  
  param_t(weight_t _alpha, weight_t _tol) : alpha(_alpha), tol(_tol) {}
};

template <typename weight_t>
struct result_t {
  weight_t* p;
  result_t(weight_t* _p)
      : p(_p) {}
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

  thrust::device_vector<weight_t> plast;    // pagerank values from previous iteration
  thrust::device_vector<weight_t> iweights; // alpha * 1 / (sum of outgoing weights) -- used to determine out of mass spread from src to dst

  void init() {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    plast.resize(n_vertices);
    iweights.resize(n_vertices);
  }

  void reset() {
    auto context0 = this->context->get_context(0)->stream();
    
    auto g = this->get_graph();
    
    auto n_vertices = g.get_number_of_vertices();
    auto alpha      = this->param.alpha;
    
    thrust::fill_n(thrust::cuda::par.on(context0),
      this->result.p, n_vertices, 1.0 / n_vertices);
    
    thrust::fill_n(thrust::cuda::par.on(context0),
      plast.begin(), n_vertices, 0);
    
    auto get_weight = [=] __device__ (const int& i) -> weight_t {
      weight_t val = 0;
      
      edge_t start = g.get_starting_edge(i);
      edge_t end   = start + g.get_number_of_neighbors(i);
      for(edge_t offset = start; offset < end; offset++) {
        val += g.get_edge_weight(offset);
      }
      
      return alpha / val;
    };

    thrust::transform(
      thrust::cuda::par.on(context0),
      thrust::counting_iterator<vertex_t>(0), 
      thrust::counting_iterator<vertex_t>(n_vertices),
      iweights.begin(),
      get_weight
    );
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  // Use Base class constructor -- does this work? does it handle copy
  // constructor?
  using gunrock::enactor_t<problem_t>::enactor_t;

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void prepare_frontier(frontier_t<vertex_t>* f, cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto n_vertices = G.get_number_of_vertices();
    
    for (vertex_t v = 0; v < n_vertices; ++v)
      f->push_back(v);
  }

  void loop(cuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();
    auto p          = P->result.p;
    auto plast      = P->plast.data().get();
    auto iweights   = P->iweights.data().get();
    auto alpha      = P->param.alpha;
    
    thrust::copy_n(thrust::device, p, n_vertices, plast);
    thrust::fill_n(thrust::device, p, n_vertices, (1 - alpha) / n_vertices);
    
    auto spread_op = [p, plast, iweights] __host__ __device__(
      vertex_t const& src,
      vertex_t const& dst,
      edge_t const& edge,
      weight_t const& weight
    ) -> bool {
      weight_t update = plast[src] * iweights[src] * weight;
      math::atomic::add(p + dst, update);
      return false;
    };
    
    operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                operators::advance_direction_t::forward,
                                operators::load_balance_t::merge_path>(
        G, E, spread_op, context);
    
    E->swap_frontier_buffers(); // swap back
  }
  
  virtual bool is_converged(cuda::multi_context_t& context) {
    if(this->iteration == 0) return false;
    
    auto P   = this->get_problem();
    auto G   = P->get_graph();
    auto tol = P->param.tol;
    
    auto n_vertices = G.get_number_of_vertices();
    auto p          = P->result.p;
    auto plast      = P->plast.data().get();
  
    auto abs_diff = [=] __device__ (const int& i) -> weight_t {
      return abs(p[i] - plast[i]);
    };

    float err = thrust::transform_reduce(
      thrust::cuda::par.on(context.get_context(0).stream()),
      thrust::counting_iterator<vertex_t>(0), 
      thrust::counting_iterator<vertex_t>(n_vertices),
      abs_diff,
      (weight_t)0.0,
      thrust::maximum<weight_t>()
    );
    
    return err < tol;
  }


};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type alpha,
          typename graph_t::weight_type tol,
          typename graph_t::weight_type* p      // Output
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(alpha, tol);
  result_type result(p);
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

}  // namespace sssp
}  // namespace gunrock