#include <gunrock/applications/hits.hxx>
#include "hits_problem.hxx"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <fstream>

namespace gunrock{
namespace hits{

template<typename graph_t>
result_c<graph_t>::result_c(){

}

template<typename graph_t>
void result_c<graph_t>::rank_authority(){
  this->auth_vertex.resize(this->auth.size());
  thrust::sequence(thrust::device,
                   this->auth_vertex.begin(),
                   this->auth_vertex.end(),
                   0);

  thrust::stable_sort_by_key(thrust::device,
                             this->auth.begin(),
                             this->auth.end(),
                             this->auth_vertex.begin());
}

template<typename graph_t>
void result_c<graph_t>::rank_hub(){
  this->hub_vertex.resize(this->hub.size());
  thrust::sequence(thrust::device,
                   this->hub_vertex.begin(),
                   this->hub_vertex.end(),
                   0);

  thrust::stable_sort_by_key(thrust::device,
                             this->hub.begin(),
                             this->hub.end(),
                             this->hub_vertex.begin());
}

template<typename graph_t>
void result_c<graph_t>::print_result(std::ostream& os){

  os<<"===Authority\n\n";
  for(int i = 0; i < this->auth.size(); i++){
    os<<"vertex ID: "<<this->auth_vertex[i]<<std::endl;
    os<<"authority: "<<this->auth[i]<<std::endl;
  }

  os<<"===Hub\n\n";
  for(int i = 0; i < this->hub.size(); i++){
    os<<"vertex ID: "<<this->hub_vertex[i]<<std::endl;
    os<<"authority: "<<this->hub[i]<<std::endl;
  }
}

template<typename graph_t>
void result_c<graph_t>::print_result(int max_vertices, std::ostream& os){

  int vertices = (max_vertices < this->hub.size())
            ? max_vertices
            : this->hub.size();

  os<<"===Authority\n\n";
  for(int i = 0; i < vertices; i++){
    os<<"vertex ID: "<<this->auth_vertex[i]<<std::endl;
    os<<"authority: "<<this->auth[i]<<std::endl;
  }

  os<<"===Hub\n\n";
  for(int i = 0; i < vertices; i++){
    os<<"vertex ID: "<<this->hub_vertex[i]<<std::endl;
    os<<"authority: "<<this->hub[i]<<std::endl;
  }
}

}// namespace hits
}// namespace gunrock
