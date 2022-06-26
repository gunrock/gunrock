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

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace bc {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename weight_t>
struct result_t {
  weight_t* bc_values;
  result_t(weight_t* _bc_values) : bc_values(_bc_values) {}
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

  thrust::device_vector<vertex_t> labels;
  thrust::device_vector<weight_t> deltas;
  thrust::device_vector<weight_t> sigmas;

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    labels.resize(n_vertices);
    deltas.resize(n_vertices);
    sigmas.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();

    auto g = this->get_graph();

    auto n_vertices = g.get_number_of_vertices();

    auto d_sigmas = thrust::device_pointer_cast(sigmas.data());
    auto d_labels = thrust::device_pointer_cast(labels.data());
    auto d_deltas = thrust::device_pointer_cast(deltas.data());

    thrust::fill_n(policy, d_sigmas, n_vertices, 0);
    thrust::fill_n(policy, d_labels, n_vertices, -1);
    thrust::fill_n(policy, d_deltas, n_vertices, 0);

    thrust::fill(policy, d_sigmas + this->param.single_source,
                 d_sigmas + this->param.single_source + 1, 1);
    thrust::fill(policy, d_labels + this->param.single_source,
                 d_labels + this->param.single_source + 1, 0);
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
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  bool forward = true;
  bool backward = true;
  std::size_t depth = 0;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    this->frontiers[0].push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();

    auto single_source = P->param.single_source;
    auto sigmas = P->sigmas.data().get();
    auto labels = P->labels.data().get();
    auto deltas = P->deltas.data().get();

    auto bc_values = P->result.bc_values;

    auto policy = context.get_context(0)->execution_policy();

    if (forward) {
      // Run advance
      auto forward_op = [sigmas, labels] __host__ __device__(
                            vertex_t const& src, vertex_t const& dst,
                            edge_t const& edge,
                            weight_t const& weight) -> bool {
        auto new_label = labels[src] + 1;
        auto old_label = math::atomic::cas(labels + dst, -1, new_label);

        if ((old_label != -1) && (new_label != old_label))
          return false;

        math::atomic::add(sigmas + dst, sigmas[src]);
        return old_label == -1;
      };

      while (true) {
        auto in_frontier = &(this->frontiers[this->depth]);
        auto out_frontier = &(this->frontiers[this->depth + 1]);

        operators::advance::execute<operators::load_balance_t::merge_path,
                                    operators::advance_direction_t::forward,
                                    operators::advance_io_type_t::vertices,
                                    operators::advance_io_type_t::vertices>(
            G, forward_op, in_frontier, out_frontier, E->scanned_work_domain,
            context);

        this->depth++;
        if (is_forward_converged(context))
          break;
      }

    } else {
      // Run advance
      auto backward_op =
          [sigmas, labels, bc_values, deltas, single_source] __host__
          __device__(vertex_t const& src, vertex_t const& dst,
                     edge_t const& edge, weight_t const& weight) -> bool {
        if (src == single_source)
          return false;

        auto s_label = labels[src];
        auto d_label = labels[dst];
        if (s_label + 1 != d_label)
          return false;

        auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
        math::atomic::add(deltas + src, update);
        math::atomic::add(bc_values + src, 0.5f * update);  // scaled output

        return false;
      };

      while (true) {
        auto in_frontier = &(this->frontiers[this->depth]);
        auto out_frontier = &(this->frontiers[this->depth + 1]);

        operators::advance::execute<operators::load_balance_t::merge_path,
                                    operators::advance_direction_t::forward,
                                    operators::advance_io_type_t::vertices,
                                    operators::advance_io_type_t::none>(
            G, backward_op, in_frontier, out_frontier, E->scanned_work_domain,
            context);

        this->depth--;
        if (is_backward_converged(context))
          break;
      }
    }
  }

  bool is_forward_converged(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto out_frontier = &(this->frontiers[this->depth]);
    bool forward_converged = out_frontier->is_empty();
    if (forward_converged) {
      forward = false;
      return true;
    }
    return false;
  }

  bool is_backward_converged(gcuda::multi_context_t& context) {
    if (depth == 0) {
      backward = false;
      return true;
    }

    return false;
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    return (!forward && !backward) ? true : false;
  }
};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type single_source,
          typename graph_t::weight_type* bc_values,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<weight_t>;

  param_type param(single_source);
  result_type result(bc_values);
  // </user-defined>

  // <boiler-plate>
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers management:
  enactor_properties_t props;
  props.number_of_frontier_buffers = 1000;  // XXX: hack!
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, context, props);
  return enactor.enact();
  // </boiler-plate>
}

template <typename graph_t>
float run(graph_t& G, typename graph_t::weight_type* bc_values) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  vertex_t n_vertices = G.get_number_of_vertices();
  auto d_bc_values = thrust::device_pointer_cast(bc_values);
  thrust::fill_n(thrust::device, d_bc_values, n_vertices, (weight_t)0);

  auto f = [&](std::size_t job_idx) -> float {
    return bc::run(G, (vertex_t)job_idx, bc_values);
  };

  std::size_t n_jobs = n_vertices;
  thrust::host_vector<float> total_elapsed(1);
  operators::batch::execute(f, n_jobs, total_elapsed.data());

  return total_elapsed[0];
}

}  // namespace bc
}  // namespace gunrock