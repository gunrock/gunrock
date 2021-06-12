#include <gunrock/framework/problem.hxx>

#pragma once

namespace gunrock{
namespace hits{

template <typename graph_t>
class problem_t: gunrock::problem_t<graph_t> {

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

private:
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

  // Functor for deviding each element with sum and taking the sqrt
  template<typename T>
  struct op{
    __device__ __host__
    T operator()(const T& x) const {
      return sqrt(x/this->sum);
      }
  };


public:
  problem_t(graph_t& G,
            std::shared_ptr<cuda::multi_context_t> _context,
            int max_iterations);

  void init() override;
  void reset() override;

  bool is_converged();
  void update_iterator();

  // Swap the auth_curr <-> auth_next
  // and hub_curr <-> hub_next
  __device__
  void swap_buffer();

  //qqq how to sync?
  __device__
  void update_auth(int dest_pos, int source_pos);

  __device__
  void update_hub(int dest_pos, int source_pos);

  __device__
  void norm_auth();

  __device__
  void norm_hub();

  thrust::device_vector<float>
  get_auth(){
    return this->auth_curr;
  }

  thrust::device_vector<float>
  get_hub(){
    return this->hub_curr;
  }

};// class problem_c

}// namespace hits
}// namespace gunrock
