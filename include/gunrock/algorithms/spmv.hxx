/**
 * @file spmv.hxx
 * @author Daniel Loran (dcloran@ucdavis.edu)
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse-Matrix Vector Multiplication.
 * @version 0.1
 * @date 2021-11-05
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace spmv {

template <typename weight_t>
struct param_t {
  weight_t* x;
  param_t(weight_t* _x) : x(_x) {}
};

template <typename weight_t>
struct result_t {
  weight_t* y;
  result_t(weight_t* _y) : y(_y) {}
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
  // </boilerplate>

  thrust::device_vector<weight_t> dummy;

  void init() override {}

  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();
    auto g = this->get_graph();
    thrust::fill_n(policy, this->result.y, g.get_number_of_vertices(), 0.0f);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(gcuda::multi_context_t& context) override {
    // TODO: Use a parameter (enum) to select between the two:
    // Maybe use the existing advance_direction_t enum.
    pull(context);
  }

  void push(gcuda::multi_context_t& context) {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto y = P->result.y;
    auto x = P->param.x;

    auto spmv = [=] __host__ __device__(
                    vertex_t const& source,    // ... source (row index)
                    vertex_t const& neighbor,  // neighbor (column index)
                    edge_t const& edge,        // edge (row ↦ column)
                    weight_t const& weight     // weight (nonzero).
                    ) -> bool {
      // y[row index] += weight * x[column index]
      math::atomic::add(&(y[source]), weight * x[neighbor]);
      return false;  // ignored.
    };

    // Perform advance on the above lambda-op
    operators::advance::execute<
        operators::load_balance_t::block_mapped,
        operators::advance_direction_t::forward,  // direction (backward for
                                                  // transpose)
        operators::advance_io_type_t::graph,      // entire graph as input
        operators::advance_io_type_t::none>(      // no output frontier needed
        G, E, spmv, context);
  }

  void pull(gcuda::multi_context_t& context) {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto y = P->result.y;
    auto x = P->param.x;

    auto spmv = [=] __host__ __device__(edge_t edge) {
      weight_t weight = G.get_edge_weight(edge);
      vertex_t neighbor = G.get_destination_vertex(edge);
      return weight * thread::load(&x[neighbor]);
    };

    // Perform neighbor-reduce with plus arithmetic operator.
    auto plus_t = [] __host__ __device__(weight_t a, weight_t b) {
      return a + b;
    };
    operators::neighborreduce::execute(G, E, y, spmv, plus_t, weight_t(0),
                                       context);
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    return this->iteration == 0 ? false : true;
  }
};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type* x,  // Input vector
          typename graph_t::weight_type* y,  // Output vector
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(x);
  result_type result(y);
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

}  // namespace spmv
}  // namespace gunrock