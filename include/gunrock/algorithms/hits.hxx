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
#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/device_vector.h>
#include <fstream>

#pragma once

namespace gunrock {
namespace hits {

const int default_max_iterations = 50;

template<typename graph_t>
class result_c{
     using vertex_t = typename graph_t::vertex_type;

private:

  int max_pages;
  graph_t& G;

  thrust::device_vector<float> auth;
  thrust::device_vector<float> hub;
  thrust::device_vector<vertex_t> auth_vertex;
  thrust::device_vector<vertex_t> hub_vertex;

public:

  result_c(graph_t& G);

  void rank_authority();
  void rank_hub();
  void print_result(std::ostream& os = std::cout);
  void print_result(int max_vertices, std::ostream& os = std::cout);

  // For internal use
  thrust::device_vector<float>
  get_auth(){
  return this->auth;
  }

  thrust::device_vector<float>
  get_hub(){
  return this->hub;
  }
};// end result_c

template<typename graph_t>
result_c<graph_t>& run(graph_t& G, int iter_times = default_max_iterations);


}// namespace hits
}// namespace gunrock
