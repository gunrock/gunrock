/**
 * @file problem.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/graph/graph.hxx>

namespace gunrock {
/**
 * @brief Inherit problem class for your custom applications' implementation.
 * Problem describes the data slice of your aglorithm, and the data can often be
 * replicated or partitioned to multiple instances (for example, in a multi-gpu
 * context). In the algorithms' problem constructor, initialize your data.
 *
 * @tparam graph_t
 * @tparam meta_t
 */
template <typename graph_t, typename meta_t, typename param_t, typename result_t>
struct problem_t {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  
  param_t* param;
  result_t* result;

  graph_t* graph_slice;
  meta_t* meta_slice;
  std::shared_ptr<cuda::multi_context_t> context;

  graph_t* get_graph_pointer() const { return graph_slice; }
  meta_t* get_meta_pointer() const { return meta_slice; }
    
  problem_t() : graph_slice(nullptr) {}
  problem_t(
    graph_t* G,
    meta_t* meta,
    param_t* _param,
    result_t* _result,
    std::shared_ptr<cuda::multi_context_t> _context
  ) : 
    graph_slice(G),
    meta_slice(meta),
    param(_param), 
    result(_result),
    context(_context) { }

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
  problem_t(const problem_t& rhs) = delete;             // Copy constructor
  problem_t& operator=(const problem_t& rhs) = delete;  // Copy assignment

};  // struct problem_t

}  // namespace gunrock