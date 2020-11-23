/**
 * @file bfs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path graph algorithm. This is where
 * we actually implement BFS using operators.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <bits/stdc++.h>

#include <gunrock/applications/application.hxx>

namespace gunrock {
namespace bfs {

template <typename meta_t>
struct param_t {
  using vertex_t = typename meta_t::vertex_type;

  vertex_t single_source;

  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename meta_t>
struct result_t {
  using vertex_t = typename meta_t::vertex_type;

  vertex_t* distances;
  vertex_t* predecessors;

  result_t(vertex_t* _distances, vertex_t* _predecessors)
      : distances(_distances), predecessors(_predecessors) {}
};

template <typename graph_t,
          typename meta_t,
          typename param_t,
          typename result_t>
struct problem_t : gunrock::problem_t<graph_t, meta_t, param_t, result_t> {
  // Use Base class constructor -- does this work? does it handle copy
  // constructor?
  using gunrock::problem_t<graph_t, meta_t, param_t, result_t>::problem_t;

  using vertex_t = typename meta_t::vertex_type;
  using edge_t = typename meta_t::edge_type;
  using weight_t = typename meta_t::weight_type;

  thrust::device_vector<vertex_t> visited;

  void init() {}

  void reset() {
    auto n_vertices = this->get_meta_pointer()->get_number_of_vertices();

    auto d_distances = thrust::device_pointer_cast(this->result->distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices, -1);

    thrust::fill(thrust::device, d_distances + this->param->single_source,
                 d_distances + this->param->single_source + 1, 0);
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

  void prepare_frontier(cuda::standard_context_t* context) override {
    auto P = this->get_problem_pointer();
    auto f = this->get_active_frontier_buffer();
    f->push_back(P->param->single_source);
  }

  void loop(cuda::standard_context_t* context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem_pointer();
    auto G = P->get_graph_pointer();

    auto single_source = P->param->single_source;
    auto distances = P->result->distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;

    auto search = [distances, single_source, iteration] __host__ __device__(
                      vertex_t const& source,    // ... source
                      vertex_t const& neighbor,  // neighbor
                      edge_t const& edge,        // edge
                      weight_t const& weight     // weight (tuple).
                      ) -> bool {
      if (distances[neighbor] != -1)
        return false;
      else
        return (math::atomic::cas(&distances[neighbor], -1, iteration + 1) ==
                -1);
    };

    auto remove_visited =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      return (vertex != std::numeric_limits<vertex_t>::max());
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                operators::advance_direction_t::forward,
                                operators::load_balance_t::merge_path>(
        G, E, search, context);

    // Execute filter operator on the provided lambda
    operators::filter::execute<operators::filter_type_t::uniquify>(
        G, E, remove_visited, context);
  }

};  // struct enactor_t

template <typename graph_t, typename meta_t>
float run(std::shared_ptr<graph_t>& G,
          std::shared_ptr<meta_t>& meta,
          typename meta_t::vertex_type& single_source,  // Parameter
          typename meta_t::vertex_type* distances,      // Output
          typename meta_t::vertex_type* predecessors    // Output
) {
  // <user-defined>
  param_t<meta_t> param(single_source);
  result_t<meta_t> result(distances, predecessors);
  // </user-defined>

  auto multi_context =
      std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));

  using problem_type =
      problem_t<graph_t, meta_t, param_t<meta_t>, result_t<meta_t>>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G.get(), meta.get(), &param, &result, multi_context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, multi_context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace bfs
}  // namespace gunrock