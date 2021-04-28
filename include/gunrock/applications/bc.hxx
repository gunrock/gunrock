/**
 * @file bc.hxx
 * @author Ben Johnson (bkj.322@gmail.com)
 * @brief Betweeness Centrality
 * @version 0.1
 * @date 2021-04
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/applications/application.hxx>

namespace gunrock {
namespace bc {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;

  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename weight_t>
struct result_t {
  weight_t* sigmas;
  weight_t* bc_values;
  result_t(weight_t* _sigmas, weight_t* _bc_values) : sigmas(_sigmas), bc_values(_bc_values) {}
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

  thrust::device_vector<vertex_t> labels;
  thrust::device_vector<weight_t> deltas;
  
  thrust::device_vector<vertex_t> my_frontiers;
  thrust::host_vector<vertex_t> offsets;

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    labels.resize(n_vertices);
    deltas.resize(n_vertices);
    my_frontiers.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();

    auto g = this->get_graph();

    auto n_vertices = g.get_number_of_vertices();
    
    auto d_sigmas    = thrust::device_pointer_cast(this->result.sigmas);
    auto d_bc_values = thrust::device_pointer_cast(this->result.bc_values);
    auto d_labels    = thrust::device_pointer_cast(labels.data());
    auto d_deltas    = thrust::device_pointer_cast(deltas.data());
    auto d_my_frontiers = thrust::device_pointer_cast(my_frontiers.data());
    
    thrust::fill_n(policy, d_sigmas,    n_vertices, 0);
    thrust::fill_n(policy, d_bc_values, n_vertices, 0);
    thrust::fill_n(policy, d_labels,    n_vertices, -1);
    thrust::fill_n(policy, d_deltas,    n_vertices, 0);

    thrust::fill(policy, d_sigmas + this->param.single_source, d_sigmas + this->param.single_source + 1, 1);
    thrust::fill(policy, d_labels + this->param.single_source, d_labels + this->param.single_source + 1, 0);

    offsets.clear();
    offsets.push_back(0);
    thrust::fill_n(policy, d_my_frontiers, n_vertices, -1);
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

  bool forward = true;
  edge_t depth = 0;

  void prepare_frontier(frontier_t<vertex_t>* f, cuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(cuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();

    auto single_source = P->param.single_source;
    auto sigmas        = P->result.sigmas;
    auto labels        = P->labels.data().get();
    auto bc_values     = P->result.bc_values;
    auto deltas        = P->deltas.data().get();
    auto offsets       = P->offsets;
    auto my_frontiers  = P->my_frontiers.data().get();
    auto depth_        = depth;

    auto policy = context.get_context(0)->execution_policy();

    if(forward) {
      // This should be ~ identical to original gunrock
      auto forward_op = [sigmas, labels] __host__ __device__(
        vertex_t const& src, vertex_t const& dst,
        edge_t const& edge, weight_t const& weight) -> bool {
        
        auto new_label = labels[src] + 1;
        auto old_label = math::atomic::cas(labels + dst, -1, new_label);
        
        if((old_label != -1) && (new_label != old_label)) return false;
        
        math::atomic::add(sigmas + dst, sigmas[src]);
        return old_label == -1;
      };

      operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                  operators::advance_direction_t::forward,
                                  operators::load_balance_t::merge_path>(
          G, E, forward_op, context);
      
      this->depth++;
      
      // >>
      // std::cout << "forward" << std::endl;
      
      

      auto predicate = [=] __host__ __device__(vertex_t const& i) -> bool {
        return i != -1;
      };

      auto output = P->my_frontiers.begin() + offsets.back();
      auto ptr = thrust::copy_if(
        policy,
        this->active_frontier->begin(),
        this->active_frontier->end(),
        output,
        predicate
      );

      auto new_offset = thrust::distance(P->my_frontiers.begin(), ptr);
      P->offsets.push_back(new_offset);
      
      // // >>
      // // Logging
      // std::cout << "my_frontiers:" << std::endl;
      // thrust::host_vector<vertex_t> tmp;
      // tmp = P->my_frontiers;
      // thrust::copy(tmp.begin(), tmp.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
      // std::cout << std::endl;
      
      // std::cout << "offsets:" << std::endl;
      // thrust::copy(P->offsets.begin(), P->offsets.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
      // std::cout << std::endl;
      // // <<
      
    } else {
      
      // std::cout << "backward" << std::endl;

      // std::cout << "offsets:" << std::endl;
      // thrust::copy(P->offsets.begin(), P->offsets.end(), std::ostream_iterator<vertex_t>(std::cout, " "));
      // std::cout << std::endl;
      
      auto offset1 = P->offsets.back();
      P->offsets.pop_back();
      auto offset0 = P->offsets.back();
      this->active_frontier->set_number_of_elements(offset1 - offset0);
      
      thrust::copy(
        policy,
        P->my_frontiers.begin() + offset0,
        P->my_frontiers.begin() + offset1,
        this->active_frontier->begin()
      );
      
      // this->active_frontier->print();
      
      auto backward_op = [sigmas, labels, bc_values, deltas, single_source, depth_] __host__ __device__(
        vertex_t const& src, vertex_t const& dst,
        edge_t const& edge, weight_t const& weight) -> bool {
        
        if(src == single_source) return false;
        
        auto s_label = labels[src];
        if(s_label != depth_)  return false;
        
        auto d_label = labels[dst];
        if(s_label + 1 != d_label) return false;
        
        auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
        math::atomic::add(deltas + src, update);
        math::atomic::add(bc_values + src, update);
        
        return false;
      };
            
      operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                  operators::advance_direction_t::forward,
                                  operators::load_balance_t::merge_path>(
          G, E, backward_op, context, false);
      
      this->depth--;
    }
  }

  virtual bool is_converged(cuda::multi_context_t& context) {

    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    
    auto n_vertices = G.get_number_of_vertices();
    
    // XXX: `bc` is a two-phase algorithm -- is there a better way to support this in the API?
    if(forward) {
      bool forward_converged = this->active_frontier->is_empty();
      if(forward_converged) {
        forward = false;
        // this->active_frontier->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());
      }
      
      return false;
      
    } else {
      if(depth == 0) { // || this->active_frontier->is_empty()) {
        // XXX: "final operation" -- is there a better way to support these kinds of things in the API?
        auto policy     = this->context->get_context(0)->execution_policy();
        auto bc_values  = P->result.bc_values;
        
        auto scale = [bc_values] __host__ __device__(weight_t& val) -> weight_t {
          return 0.5 * val;
        };
        thrust::transform(policy, bc_values, bc_values + n_vertices, bc_values, scale);
        
        return true;
      } else {
        return false;
      }
    }
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type single_source,
          typename graph_t::weight_type* sigmas,
          typename graph_t::weight_type* bc_values
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<weight_t>;

  param_type param(single_source);
  result_type result(sigmas, bc_values);
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
  
  return -1;
}

}  // namespace pr
}  // namespace gunrock