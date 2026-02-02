#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace my_sssp {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  result_t(weight_t* _distances) : distances(_distances) {}
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

  thrust::device_vector<vertex_t> visited;

  void init() {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    visited.resize(n_vertices);
  }

  void reset() {
    auto g = this->get_graph();
    auto distances = this->result.distances;
    auto n_vertices = g.get_number_of_vertices();

    thrust::fill(thrust::device, 
                 distances + 0, 
                 distances + n_vertices,
                 std::numeric_limits<weight_t>::max());

    thrust::fill(thrust::device,
                 distances + this->param.single_source,
                 distances + this->param.single_source + 1,
                 0);

    thrust::fill(thrust::device, visited.begin(), visited.end(), -1);
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
  using frontier_t = typename gunrock::enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f, gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();
    auto iteration = this->iteration;

    auto shortest_path = [distances] __host__ __device__(
        vertex_t const& source,
        vertex_t const& neighbor,
        edge_t const& edge,
        weight_t const& weight) -> bool {
      weight_t new_dist = distances[source] + weight;
      weight_t old_dist = math::atomic::min(distances + neighbor, new_dist);
      return new_dist < old_dist;
    };

    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::vertices,
                                operators::advance_io_type_t::vertices>(
        G, E, shortest_path, context);

    auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
        vertex_t const& vertex) -> bool {
      if (G.get_number_of_neighbors(vertex) == 0)
        return false;
      if (visited[vertex] == iteration)
        return false;
      visited[vertex] = iteration;
      return true;
    };

    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, remove_completed_paths, context);
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    return this->active_frontier->is_empty();
  }
};

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type single_source,
          typename graph_t::weight_type* distances,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {
  
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  param_type param(single_source);
  result_type result(distances);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace my_sssp
}  // namespace gunrock