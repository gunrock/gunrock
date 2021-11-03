#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/util/math.hxx>

namespace gunrock {
namespace spmv {

// I can't figure out any params the user would provide
template <typename weight_t>
struct param_t {
  weight_t* x;
  param_t(weight_t* _x) : x(_x) {}
};

//result_t looks exactly like a matrix, since the user will get a matrix after mtx vector mult
template <typename weight_t>
struct result_t {
    weight_t* y;
    result_t(weight_t* _y) : y(_y) {}
};

// <boilerplate>
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(
    graph_t& G,
    param_type& _param,
    result_type& _result,
    std::shared_ptr<cuda::multi_context_t> _context
  ) : gunrock::problem_t<graph_t>(G, _context), param(_param), result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
// </boilerplate>

  void init() override {}

  // `reset` function, described above.  Should be called
  void reset() override {
    auto G = this->get_graph();
    auto y = this->result.y;
    thrust::fill(
      thrust::device, 
      y,
      y + G.get_number_of_vertices(),
      0
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

  void prepare_frontier(frontier_t<vertex_t>* f,
                        cuda::multi_context_t& context) override {}

  void loop(cuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto y = this->result.y;
    auto x = this->param.x;

    auto spmv = [=] __host__ __device__(
                             vertex_t const& source,     // ... source (row index)
                             vertex_t const& neighbor,   // neighbor (column index)
                             edge_t const& edge,         // edge (row ↦ column)
                             weight_t const& weight      // weight (nonzero).
                             ) -> bool {
      // y[row index] += weight * x[column index]
      math::atomic::add(&(y[source]), weight * x[neighbor]); 
      return false; // ignored.
    };

    // Perform advance on the above lambda-op
    operators::advance::execute<operators::load_balance_t::block_mapped,    // flavor of load-balancing schedule
                                operators::advance_direction_t::forward,    // direction (backward for transpose)
                                operators::advance_io_type_t::graph,        // entire graph as input
                                operators::advance_io_type_t::none>(        // no output frontier needed
          G, E, spmv, context);
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    if (this->iteration == 0){
      return false;
    }
    return true;
    }
  };  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type* x, // Input vector
          typename graph_t::weight_type* y // Output
) {

  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(x);
  result_type result(y);
  // </user-defined>
  
  // <boiler-plate>
  auto multi_context =
      std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, multi_context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, multi_context, props);
  return enactor.enact();
  // </boiler-plate>
}

} // namespace spmv
} // namespace gunrock