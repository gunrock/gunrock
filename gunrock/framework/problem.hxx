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
 */
template <typename graph_t>
struct problem_t {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  graph_t graph_slice;
  std::shared_ptr<cuda::multi_context_t> context;

  auto get_graph() { return graph_slice; }

  problem_t() : graph_slice(nullptr) {}

  problem_t(graph_t& G, std::shared_ptr<cuda::multi_context_t> _context)
      : graph_slice(G), context(_context) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
  problem_t(const problem_t& rhs) = delete;             // Copy constructor
  problem_t& operator=(const problem_t& rhs) = delete;  // Copy assignment

};  // struct problem_t

}  // namespace gunrock