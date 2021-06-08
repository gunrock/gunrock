#include <gunrock/util/math.hxx>
#include <gunrock/applications/hits.hxx>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

// using namespace gunrock;
// using namespace hits;
namespace gunrock{
namespace hits{

template <typename graph_t>
problem_t<graph_t>::problem_t(graph_t& G,
            std::shared_ptr<cuda::multi_context_t> _context,
            int max_iterations)
      : gunrock::problem_t<graph_t>(G, _context),
        max_iterations(max_iterations){
  vertex_num = G.get_number_of_vertices;
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

template <typename graph_t>
void problem_t<graph_t>::init(){
  // Why anthor graph?
  auto g = this->get_graph();
}

template <typename graph_t>
void problem_t<graph_t>::reset(){
  auto g = this->get_graph();
  auto n_vertices = g.get_number_of_vertices();

  auto context = this->get_single_context();
  auto policy = context->execution_policy();

  auto d_distances = thrust::device_pointer_cast(this->result.distances);
  thrust::fill(policy, d_distances + 0, d_distances + n_vertices,
               std::numeric_limits<weight_t>::max());

  thrust::fill(policy, d_distances + this->param.single_source,
               d_distances + this->param.single_source + 1, 0);
}

template <typename graph_t>
bool problem_t<graph_t>::is_converged(){
  if(this->max_iterations <= iterator){
    // qqq notify exceeding
    return true;
  }
  // qqq omit policy
  else if(thrust::equal(this->auth_curr.begin(),
          this->auth_curr.end(),
          this->auth_next.begin())){
    return true;
  }
  else if(thrust::equal(this->hub_curr.begin(),
          this->hub_curr.end(),
          this->hub_next.begin())){
    return true;
  }
  else{
    return false;
  }
}

template <typename graph_t>
void problem_t<graph_t>::update_iterator(){
  ++this->iterator;
}

template <typename graph_t>
__device__
void problem_t<graph_t>::swap_buffer(){
  thrust::swap(auth_curr, auth_next);
  thrust::swap(hub_curr, hub_next);
}

template <typename graph_t>
__device__
void problem_t<graph_t>::update_auth(int dest_pos, int source_pos){
  gunrock::math::atomic::add((&this->auth_next[dest_pos]).get(),
                             (float)this->hub_curr[source_pos]);
}

template <typename graph_t>
__device__
void problem_t<graph_t>::update_hub(int dest_pos, int source_pos){
  gunrock::math::atomic::add((&this->hub_next[dest_pos]).get(),
                             (float)this->auth_curr[source_pos]);
}

template <typename graph_t>
__device__
void problem_t<graph_t>::norm_auth(){
  auto policy = this->get_single_context(0)->execution_policy();

  thrust::for_each(policy,
                    this->auth_next.begin(),
                    this->auth_next.end(),
                    thrust::square<float>());
  this->sum =
  thrust::reduce(policy,
                 this->auth_next.begin(),
                 this->auth_next.end());

  thrust::for_each(policy,
                   this->auth_next.begin(),
                   this->auth_next.end(),
                   op<float>());
}

template <typename graph_t>
__device__
void problem_t<graph_t>::norm_hub(){
  auto policy = this->get_single_context(0)->execution_policy();

  thrust::for_each(policy,
                    this->hub_next.begin(),
                    this->hub_next.end(),
                    thrust::square<float>());
  this->sum =
  thrust::reduce(policy,
                 this->hub_next.begin(),
                 this->hub_next.end());

  thrust::for_each(policy,
                   this->hub_next.begin(),
                   this->hub_next.end(),
                   op<float>());
}


}// namespace hits
}// namespace gunrock
