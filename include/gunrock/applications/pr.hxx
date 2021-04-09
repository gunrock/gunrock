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

namespace gunrock {
namespace pr {

struct param_t {
  // No parameters
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

  thrust::device_vector<weight_t> pnext;

  void init() {
    auto g = this->get_graph();
    // auto n_vertices = g.get_number_of_vertices();
    // pnext.resize(n_vertices);
  }

  void reset() {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    // thrust::fill_n(thrust::device, this->result.p, n_vertices, 1.0 / n_vertices);
    // thrust::fill_n(thrust::device, pnext.begin(), n_vertices, 0);    
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

    // XXX: Find a better way to initialize the frontier to all nodes
    // for (vertex_t v = 0; v < n_vertices; ++v) {
      f->push_back(0);
    // }
  }

  void loop(cuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();
    // auto p          = P->result.p;
    // auto pnext      = P->pnext.data().get();

    auto iteration = this->iteration;
    printf("ok\n");
    
    if(iteration != 0) {
      auto filter_op = [] __host__ __device__ (
        vertex_t const& src,
        vertex_t const& dst,
        edge_t const& edge,
        weight_t const& weight
      ) -> bool {
        
        // weight_t delta     = 0.85;
        // weight_t threshold = 0.01;
        
        // weight_t old_val = p[dst];
        // weight_t new_val = (1 - delta) + delta * pnext[dst];
        
        // if(G.get_number_of_neighbors(dst) > 0) {
        //   new_val /= G.get_number_of_neighbors(dst);
        // }
        
        // if(new_val > 999) {
        //   new_val = 0;
        // }
        
        // p[dst] = new_val;
        
        // return fabs(new_val - old_val) > (threshold * old_val);
        return false;
      };
      
      operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                  operators::advance_direction_t::forward,
                                  operators::load_balance_t::thread_mapped>(
          G, E, filter_op, context);
      
      // thrust::fill_n(thrust::device, pnext, n_vertices, 0);
    }
    
    auto advance_op = [] __host__ __device__ (
      vertex_t const& src,
      vertex_t const& dst,
      edge_t const& edge,
      weight_t const& weight
    ) -> bool {
      // weight_t val = p[src];
      // if(val < 9999) {
      //   math::atomic::add(pnext + dst, val);
      // }
      
      return true;
    };

    operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
                                operators::advance_direction_t::forward,
                                operators::load_balance_t::thread_mapped>(
        G, E, advance_op, context);

    // auto shortest_path = [distances, single_source] __host__ __device__(
    //                          vertex_t const& source,    // ... source
    //                          vertex_t const& neighbor,  // neighbor
    //                          edge_t const& edge,        // edge
    //                          weight_t const& weight     // weight (tuple).
    //                          ) -> bool {
    //   weight_t source_distance = distances[source];  // use cached::load
    //   weight_t distance_to_neighbor = source_distance + weight;

    //   // Check if the destination node has been claimed as someone's child
    //   weight_t recover_distance =
    //       math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

    //   return (distance_to_neighbor < recover_distance);
    // };

    // auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
    //                                   vertex_t const& vertex) -> bool {
    //   if (visited[vertex] == iteration)
    //     return false;

    //   visited[vertex] = iteration;
    //   return G.get_number_of_neighbors(vertex) > 0;
    // };

    // // Execute advance operator on the provided lambda
    // operators::advance::execute<operators::advance_type_t::vertex_to_vertex,
    //                             operators::advance_direction_t::forward,
    //                             operators::load_balance_t::thread_mapped>(
    //     G, E, shortest_path, context);

    // // Execute filter operator on the provided lambda
    // operators::filter::execute<operators::filter_algorithm_t::predicated>(
    //     G, E, remove_completed_paths, context);
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type* p      // Output
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t;
  using result_type = result_t<weight_t>;

  param_type param;
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