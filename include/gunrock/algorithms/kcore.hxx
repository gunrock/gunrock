/**
 * @file kcore.hxx
 * @author Afton Geil (angeil@ucdavis.edu)
 * @brief Vertex k-core decomposition algorithm.
 * @version 0.1
 * @date 2021-05-03
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/logical.h>

namespace gunrock {
namespace kcore {

template <typename vertex_t>
struct result_t {
  int* k_cores;
  result_t(int* _k_cores) : k_cores(_k_cores) {}
};

template <typename graph_t, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  result_type result;

  problem_t(graph_t& G,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context), result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<int> degrees;
  thrust::device_vector<bool> deleted;
  thrust::device_vector<bool> to_be_deleted;

  void init() {
    // Get the graph
    auto g = this->get_graph();

    // Get number of vertices from the graph
    auto n_vertices = g.get_number_of_vertices();

    // Set the size of `degrees`, `deleted`, and `to_be_deleted`
    degrees.resize(n_vertices);
    deleted.resize(n_vertices);
    to_be_deleted.resize(n_vertices);
  }

  void reset() {
    auto g = this->get_graph();

    auto k_cores = this->result.k_cores;
    auto n_vertices = g.get_number_of_vertices();

    // set `k_cores`, `deleted`, and `to_be_deleted` to 0 for all vertices
    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill(policy, k_cores + 0, k_cores + n_vertices, 0);
    thrust::fill(policy, to_be_deleted.begin(), to_be_deleted.end(), 0);

    // set initial `degrees` values to be vertices' actual degree
    // will reduce these as vertices are removed from k-cores with increasing k
    // value
    auto get_degree = [=] __device__(const int& i) -> int {
      return g.get_number_of_neighbors(i);
    };

    thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                      thrust::counting_iterator<vertex_t>(n_vertices),
                      degrees.begin(), get_degree);

    // mark zero degree vertices as deleted
    auto degrees_data = degrees.data().get();
    auto mark_zero_degrees = [=] __device__(const int& i) -> bool {
      return (degrees_data[i] == 0) ? true : false;
    };

    thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                      thrust::counting_iterator<vertex_t>(n_vertices),
                      deleted.begin(), mark_zero_degrees);
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
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    // get pointer to the problem
    auto P = this->get_problem();
    auto n_vertices = P->get_graph().get_number_of_vertices();

    // Fill the frontier with a sequence of vertices from 0 -> n_vertices.
    f->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());
  }

  // One iteration of the application
  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    // Get parameters and data structures
    auto k_cores = P->result.k_cores;
    auto degrees = P->degrees.data().get();
    auto deleted = P->deleted.data().get();
    auto to_be_deleted = P->to_be_deleted.data().get();
    auto n_vertices = G.get_number_of_vertices();
    auto f = this->get_input_frontier();

    // Get current iteration of application
    auto k = this->iteration + 1;

    // Mark vertices with degree <= k for deletion and output their neighbors
    auto advance_op = [=] __host__ __device__(
                          vertex_t const& source,    // source of edge
                          vertex_t const& neighbor,  // destination of edge
                          edge_t const& edge,        // id of edge
                          weight_t const& weight     // weight of edge
                          ) -> bool {
      if ((deleted[source] == true) || (degrees[source] > k)) {
        return false;
      } else {
        k_cores[source] = k;
        to_be_deleted[source] = true;
        if (deleted[neighbor] == true) {
          return false;
        }
        return true;
      }
    };

    // Reduce degrees of deleted vertices' neighbors
    // Check updated degree against k
    auto filter_op = [=] __host__ __device__(vertex_t const& vertex) -> bool {
      if (deleted[vertex] == true) {
        return false;
      }

      int old_degrees = math::atomic::add(&degrees[vertex], -1);
      return (old_degrees != (k + 1)) ? false : true;
    };

    while (!f->is_empty()) {
      // Execute advance operator
      operators::advance::execute<operators::load_balance_t::block_mapped>(
          G, E, advance_op, context);

      // Mark to-be-deleted vertices as deleted
      auto mark_deleted = [=] __device__(const vertex_t& v) {
        deleted[v] = (deleted[v] | to_be_deleted[v]);
      };

      operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
          G,             // graph
          mark_deleted,  // lambda function
          context        // context
      );

      // Execute filter operator
      operators::filter::execute<operators::filter_algorithm_t::predicated>(
          G, E, filter_op, context);
    }
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto n_vertices = G.get_number_of_vertices();
    auto f = this->get_input_frontier();
    auto policy = context.get_context(0)->execution_policy();

    //  Check if all vertices have been removed from graph
    bool graph_empty = thrust::all_of(
        policy, P->deleted.begin(), P->deleted.end(), thrust::identity<bool>());

    if (graph_empty) {
      printf("degeneracy = %u\n", this->iteration);
    }

    // Fill the frontier with a sequence of vertices from 0 -> n_vertices.
    f->sequence((vertex_t)0, n_vertices, context.get_context(0)->stream());

    return graph_empty;
  }
};

template <typename graph_t>
float run(graph_t& G,
          int* k_cores,  // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  // instantiate `result` template
  using result_type = result_t<int>;

  // initialize `result` w/ the appropriate parameters / data structures
  result_type result(k_cores);

  // instantiate `problem` and `enactor` templates.
  using problem_type = problem_t<graph_t, result_type>;
  using enactor_type = enactor_t<problem_type>;

  // initialize problem; call `init` and `reset` to prepare data structures
  problem_type problem(G, result, context);
  problem.init();
  problem.reset();

  // initialize enactor; call enactor, returning GPU elapsed time
  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace kcore
}  // namespace gunrock
