/**
 * @file hits.hxx
 * @author Liyidong
 * @brief Hyperlink-Induced Topic Search.
 * @version 0.1
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

const int default_max_iterations = 50;

class param_c {
 private:
  int max_iterations;

 public:
  param_c(int max_iterations) : max_iterations(max_iterations) {}
  int get_max_iterations() { return this->max_iterations; }
};  // end of param_c

template <typename graph_t>
class result_c {
  using vertex_t = typename graph_t::vertex_type;

 private:
  int max_pages;
  graph_t& G;

  thrust::device_vector<float> auth;
  thrust::device_vector<float> hub;
  thrust::device_vector<vertex_t> auth_vertex;
  thrust::device_vector<vertex_t> hub_vertex;

 public:
  result_c(graph_t& G) : G(G) {}

  void rank_authority() {
    this->auth_vertex.resize(this->auth.size());
    thrust::sequence(thrust::device, this->auth_vertex.begin(),
                     this->auth_vertex.end(), 0);
    thrust::stable_sort_by_key(thrust::device, this->auth.begin(),
                               this->auth.end(), this->auth_vertex.begin());
  }

  void rank_hub() {
    this->hub_vertex.resize(this->hub.size());
    thrust::sequence(thrust::device, this->hub_vertex.begin(),
                     this->hub_vertex.end(), 0);
    thrust::stable_sort_by_key(thrust::device, this->hub.begin(),
                               this->hub.end(), this->hub_vertex.begin());
  }

  void print_result(std::ostream& os = std::cout) {
    os << "===Authority\n\n";
    for (int i = 0; i < this->auth.size(); i++) {
      os << "vertex ID: " << this->auth_vertex[i] << std::endl;
      os << "authority: " << this->auth[i] << std::endl;
    }
    os << "===Hub\n\n";
    for (int i = 0; i < this->hub.size(); i++) {
      os << "vertex ID: " << this->hub_vertex[i] << std::endl;
      os << "authority: " << this->hub[i] << std::endl;
    }
  }

  void print_result(int max_vertices, std::ostream& os = std::cout) {
    int vertices =
        (max_vertices < this->hub.size()) ? max_vertices : this->hub.size();
    os << "===Authority\n\n";
    for (int i = 0; i < vertices; i++) {
      os << "vertex ID: " << this->auth_vertex[i] << std::endl;
      os << "authority: " << this->auth[i] << std::endl;
    }
    os << "===Hub\n\n";
    for (int i = 0; i < vertices; i++) {
      os << "vertex ID: " << this->hub_vertex[i] << std::endl;
      os << "authority: " << this->hub[i] << std::endl;
    }
  }

  // For internal use
  thrust::device_vector<float> get_auth() { return this->auth; }

  thrust::device_vector<float> get_hub() { return this->hub; }
};  // end of result_c

template <typename graph_t>
struct problem_t : gunrock::problem_t<graph_t> {
 public:
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  int iterator = 0;
  int max_iterations;

  // Sum of squares of authority and hub
  // qqq find a way to reduce mem access
  float sum = 0;

  vertex_t vertex_num = 0;

  thrust::device_vector<float> auth_curr;
  thrust::device_vector<float> hub_curr;
  thrust::device_vector<float> auth_next;
  thrust::device_vector<float> hub_next;

  // poniters to the data inside the device_vector
  float* auth_curr_p = nullptr;
  float* hub_curr_p = nullptr;
  float* auth_next_p = nullptr;
  float* hub_next_p = nullptr;

  // Functor for deviding each element with sum and taking the sqrt
  template <typename T>
  struct op {
   private:
    float sum;

   public:
    op(float sum) : sum(sum) {}
    __device__ __host__ T operator()(const T& x) const {
      return sqrt(x / this->sum);
    }
  };

  problem_t(graph_t& G,
            std::shared_ptr<gcuda::multi_context_t> _context,
            int max_iterations)
      : gunrock::problem_t<graph_t>(G, _context),
        max_iterations(max_iterations) {
    vertex_num = G.get_number_of_vertices();
    auto policy = _context->get_context(0)->execution_policy();

    auth_curr.resize(vertex_num);
    thrust::fill(policy, auth_curr.begin(), auth_curr.end(), 0);
    auth_next.resize(vertex_num);
    thrust::fill(policy, auth_next.begin(), auth_next.end(), 0);
    hub_curr.resize(vertex_num);
    thrust::fill(policy, hub_curr.begin(), hub_curr.end(), 0);
    hub_next.resize(vertex_num);
    thrust::fill(policy, hub_next.begin(), hub_next.end(), 0);
  }

  void init() override {
    auth_curr_p = auth_curr.data().get();
    hub_curr_p = hub_curr.data().get();
    auth_next_p = auth_next.data().get();
    hub_next_p = hub_next.data().get();
  }
  void reset() override {}

  bool is_converged() {
    if (this->max_iterations <= iterator) {
      // qqq notify exceeding
      return true;
    }
    // qqq omit policy
    else if (thrust::equal(this->auth_curr.begin(), this->auth_curr.end(),
                           this->auth_next.begin())) {
      return true;
    } else if (thrust::equal(this->hub_curr.begin(), this->hub_curr.end(),
                             this->hub_next.begin())) {
      return true;
    } else {
      return false;
    }
  }

  void update_iterator() { ++this->iterator; }

  // Swap the auth_curr <-> auth_next
  // and hub_curr <-> hub_next
  __device__ void swap_buffer() {
    thrust::swap(auth_curr, auth_next);
    thrust::swap(hub_curr, hub_next);
  }

  __device__ void update_auth(int dest_pos, int source_pos) {
    gunrock::math::atomic::add(&this->auth_next_p[dest_pos],
                               hub_curr_p[source_pos]);
  }

  __device__ void update_hub(int dest_pos, int source_pos) {
    gunrock::math::atomic::add(&hub_next_p[dest_pos], auth_curr_p[source_pos]);
  }

  __device__ void norm_auth() {
    auto policy = this->get_single_context(0)->execution_policy();

    thrust::for_each(policy, this->auth_next.begin(), this->auth_next.end(),
                     thrust::square<float>());
    this->sum =
        thrust::reduce(policy, this->auth_next.begin(), this->auth_next.end());

    op<float> op{this->sum};
    thrust::for_each(policy, this->auth_next.begin(), this->auth_next.end(),
                     op);
  }

  __device__ void norm_hub() {
    auto policy = this->get_single_context(0)->execution_policy();

    thrust::for_each(policy, this->hub_next.begin(), this->hub_next.end(),
                     thrust::square<float>());
    this->sum =
        thrust::reduce(policy, this->hub_next.begin(), this->hub_next.end());

    op<float> op{this->sum};
    thrust::for_each(policy, this->hub_next.begin(), this->hub_next.end(), op);
  }

  thrust::device_vector<float> get_auth() { return this->auth_curr; }

  thrust::device_vector<float> get_hub() { return this->hub_curr; }

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
    // Data slice qqq
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto update = [P] __host__ __device__(
                      vertex_t & source, vertex_t & neighbor,
                      edge_t const& edge, weight_t const& weight) -> bool {
      P->update_hub(source, neighbor);
      P->update_auth(neighbor, source);

      return true;
    };  // end of update

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::vertices>(
        G, E, update, context);

    // Normalize authority and hub
    P->norm_auth();
    P->norm_hub();

    // Swap buffer
    P->swap_buffer();

    // Update iterator
    P->update_iterator();

  }  // end of loop

  bool is_converged(gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    return P->is_converged();
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
template <typename graph_t, typename param_t, typename result_t>
float run(graph_t& G,
          param_t& param,
          result_t& result,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using problem_type = problem_t<graph_t>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, context, param.get_max_iterations());

  enactor_type enactor(&problem, context);
  auto time = enactor.enact();

  dump_result(result.get_auth(), result.get_hub(), problem.get_auth(),
              problem.get_hub());

  result.rank_authority();
  result.rank_hub();

  return time;
}

}  // namespace hits
}  // namespace gunrock
