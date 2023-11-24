/**
 * @file hits.hxx
 * @author Liyidong
 * @brief Hyperlink-Induced Topic Search.
 * @date 2021.05.06
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <fstream>

namespace gunrock {
namespace hits {

struct param_t {
  unsigned int max_iterations;
  param_t(unsigned int _max_iterations = 50)
      : max_iterations(_max_iterations) {}
};  // end of param_t

template <typename vertex_t, typename weight_t>
struct result_c {
  int max_pages;

  thrust::device_vector<weight_t> auth;
  thrust::device_vector<weight_t> hub;
  thrust::device_vector<vertex_t> auth_vertex;
  thrust::device_vector<vertex_t> hub_vertex;

  void rank_authority() {
    auth_vertex.resize(auth.size());
    thrust::sequence(thrust::device, auth_vertex.begin(), auth_vertex.end(), 0);
    thrust::stable_sort_by_key(thrust::device, auth.begin(), auth.end(),
                               auth_vertex.begin());
  }

  void rank_hub() {
    hub_vertex.resize(hub.size());
    thrust::sequence(thrust::device, hub_vertex.begin(), hub_vertex.end(), 0);
    thrust::stable_sort_by_key(thrust::device, hub.begin(), hub.end(),
                               hub_vertex.begin());
  }

  void print_result(std::ostream& os = std::cout) {
    print::head(auth, 20, "Authority Vertices");
    print::head(auth, 20, "Authority");

    print::head(hub_vertex, 20, "Hub Vertices");
    print::head(hub, 20, "Hub");
  }
};  // end of result_c

template <typename graph_t, typename param_type>
struct problem_t : gunrock::problem_t<graph_t> {
 public:
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  param_type param;

  thrust::device_vector<weight_t> auth_curr;
  thrust::device_vector<weight_t> hub_curr;
  thrust::device_vector<weight_t> auth_next;
  thrust::device_vector<weight_t> hub_next;

  // poniters to the data inside the device_vector
  weight_t* auth_curr_p = nullptr;
  weight_t* hub_curr_p = nullptr;
  weight_t* auth_next_p = nullptr;
  weight_t* hub_next_p = nullptr;

  problem_t(graph_t& G,
            std::shared_ptr<gcuda::multi_context_t> _context,
            param_type& _param)
      : gunrock::problem_t<graph_t>(G, _context), param(_param) {}

  void init() override {
    vertex_t n_vertices = this->get_graph().get_number_of_vertices();
    auth_curr.resize(n_vertices);
    auth_next.resize(n_vertices);
    hub_curr.resize(n_vertices);
    hub_next.resize(n_vertices);

    auth_curr_p = auth_curr.data().get();
    hub_curr_p = hub_curr.data().get();
    auth_next_p = auth_next.data().get();
    hub_next_p = hub_next.data().get();
  }
  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();

    thrust::fill(policy, auth_curr.begin(), auth_curr.end(), 0);
    thrust::fill(policy, auth_next.begin(), auth_next.end(), 0);
    thrust::fill(policy, hub_curr.begin(), hub_curr.end(), 0);
    thrust::fill(policy, hub_next.begin(), hub_next.end(), 0);
  }
};  // end of problem_c

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(gcuda::multi_context_t& context) override {
    auto policy = context.get_context(0)->execution_policy();

    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto& auth = P->auth_next;
    auto& hub = P->hub_next;

    auto auth_curr_p = P->auth_curr_p;
    auto auth_next_p = P->auth_next_p;

    auto hub_curr_p = P->hub_curr_p;
    auto hub_next_p = P->hub_next_p;

    auto update = [=] __host__ __device__(
                      vertex_t & source, vertex_t & neighbor,
                      edge_t const& edge, weight_t const& weight) -> bool {
      math::atomic::add(&hub_next_p[source], auth_curr_p[neighbor]);
      math::atomic::add(&auth_next_p[neighbor], hub_curr_p[source]);
      return true;
    };  // end of update

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::vertices>(
        G, E, update, context);

    // Normalize authority
    weight_t sum = 0;
    thrust::for_each(policy, auth.begin(), auth.end(),
                     thrust::square<weight_t>());
    sum = thrust::reduce(policy, auth.begin(), auth.end());
    thrust::for_each(policy, auth.begin(), auth.end(),
                     [=] __device__(const weight_t& x) -> weight_t {
                       return sqrt(x / sum);
                     });

    // Normalize hub
    thrust::for_each(policy, hub.begin(), hub.end(),
                     thrust::square<weight_t>());
    sum = thrust::reduce(policy, hub.begin(), hub.end());
    thrust::for_each(policy, hub.begin(), hub.end(),
                     [=] __device__(const weight_t& x) -> weight_t {
                       return sqrt(x / sum);
                     });

    // Swap buffer
    thrust::swap(P->auth_curr, P->auth_next);
    thrust::swap(P->hub_curr, P->hub_next);

  }  // end of loop

  bool is_converged(gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto iteration = this->iteration;
    if (P->param.max_iterations <= iteration) {
      return true;
    } else if (thrust::equal(P->auth_curr.begin(), P->auth_curr.end(),
                             P->auth_next.begin())) {
      return true;
    } else if (thrust::equal(P->hub_curr.begin(), P->hub_curr.end(),
                             P->hub_next.begin())) {
      return true;
    } else {
      return false;
    }
  }

};  // end of enactor_t

template <typename ForwardIterator>
void dump_result(ForwardIterator auth_dest,
                 ForwardIterator hub_dest,
                 ForwardIterator auth_src,
                 ForwardIterator hub_src) {
  thrust::swap(auth_dest, auth_src);
  thrust::swap(hub_dest, hub_src);
}

// qqq get rid of template for better control
template <typename graph_t, typename result_t>
float run(graph_t& G,
          unsigned int max_iterations,
          result_t& result,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using problem_type = problem_t<graph_t, param_t>;
  using enactor_type = enactor_t<problem_type>;

  param_t param(max_iterations);

  problem_type problem(G, context, param);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  auto time = enactor.enact();

  result.auth = problem.auth_curr;
  result.hub = problem.hub_curr;

  result.rank_authority();
  result.rank_hub();

  return time;
}

}  // namespace hits
}  // namespace gunrock
